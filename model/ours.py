"""
Geometry-Detail Collaborative Network (GDCN) for remote sensing image super-resolution.

This implementation replaces the original Mamba/SS2D reconstruction trunk with
the two modules defined in the manuscript:

1) Geometry-Regularized Prototype Transport Module (GRPTM)
2) Structure-Conditioned Detail Dictionary Refinement Module (SDDRM)

The network follows:
    shallow feature extraction -> stacked GDCBs -> global residual fusion -> upsampling

Compatible with BasicSR ARCH_REGISTRY when BasicSR is installed.
"""

import math
from typing import Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from basicsr.utils.registry import ARCH_REGISTRY
except ImportError:
    class _FallbackRegistry:
        def register(self):
            def decorator(cls):
                return cls
            return decorator
    ARCH_REGISTRY = _FallbackRegistry()

try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:
    class PyTorchModelHubMixin:
        pass


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, std=std)


class Upsample(nn.Sequential):
    """Multi-step pixel-shuffle upsampling head."""

    def __init__(self, scale: int, num_feat: int) -> None:
        layers = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                layers.extend([
                    nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
                    nn.PixelShuffle(2),
                ])
        elif scale == 3:
            layers.extend([
                nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1),
                nn.PixelShuffle(3),
            ])
        else:
            raise ValueError(f"Unsupported scale {scale}. Use powers of two or 3.")
        super().__init__(*layers)


class UpsampleOneStep(nn.Sequential):
    """Single-step pixel-shuffle head for lightweight SR."""

    def __init__(self, scale: int, num_feat: int, num_out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale),
        )


class GRPTM(nn.Module):
    r"""Geometry-Regularized Prototype Transport Module.

    It realizes the manuscript formulation:
        directional basis aggregation -> compact prototype abstraction ->
        geometry-regularized transport -> pixel-to-prototype reassembly ->
        edge-aware residual injection.

    Args:
        dim: Input/output feature channels.
        reduced_dim: Compact channel dimension C_r.
        num_prototypes: Number of structural prototypes K.
        geom_dim: Dimension C_g of the prototype-level geometry descriptor.
        transport_dim: Prototype transport dimension d; defaults to reduced_dim.
        tau: Temperature for prototype assignment.
        sigma_g: Scaling factor in the geometry affinity.
        lambda_g: Strength of Laplacian geometry regularization.
    """

    def __init__(
        self,
        dim: int = 64,
        reduced_dim: int = 32,
        num_prototypes: int = 16,
        geom_dim: int = 16,
        transport_dim: Optional[int] = None,
        tau: float = 1.0,
        sigma_g: float = 1.0,
        lambda_g: float = 0.1,
    ) -> None:
        super().__init__()
        if reduced_dim <= 0 or num_prototypes <= 0:
            raise ValueError("reduced_dim and num_prototypes must be positive.")
        self.dim = dim
        self.reduced_dim = reduced_dim
        self.num_prototypes = num_prototypes
        self.transport_dim = transport_dim or reduced_dim
        self.tau = float(tau)
        self.sigma_g = float(sigma_g)
        self.lambda_g = float(lambda_g)

        # psi(.) in Eq. (GRPTM basis / edge prior)
        self.psi = nn.Sequential(
            nn.Conv2d(dim, reduced_dim, 1, 1, 0),
            nn.GELU(),
        )

        # Directional basis operators: horizontal, vertical, dilated isotropic.
        self.directional_bases = nn.ModuleList([
            nn.Conv2d(
                reduced_dim, reduced_dim, kernel_size=(1, 3), padding=(0, 1),
                groups=reduced_dim, bias=True
            ),
            nn.Conv2d(
                reduced_dim, reduced_dim, kernel_size=(3, 1), padding=(1, 0),
                groups=reduced_dim, bias=True
            ),
            nn.Conv2d(
                reduced_dim, reduced_dim, kernel_size=3, padding=2, dilation=2,
                groups=reduced_dim, bias=True
            ),
        ])
        self.router = nn.Conv2d(reduced_dim, len(self.directional_bases), 1, 1, 0)
        self.geometry_proj = nn.Conv2d(
            len(self.directional_bases) * reduced_dim, geom_dim, 1, 1, 0
        )

        # W_p is represented by learnable prototype generator rows.
        self.prototype_generator = nn.Parameter(
            torch.empty(num_prototypes, reduced_dim)
        )

        self.q_proj = nn.Linear(reduced_dim, self.transport_dim, bias=False)
        self.k_proj = nn.Linear(reduced_dim, self.transport_dim, bias=False)
        self.v_proj = nn.Linear(reduced_dim, self.transport_dim, bias=False)
        self.location_proj = nn.Linear(reduced_dim, self.transport_dim, bias=False)

        # phi_e and phi_o in the edge-aware residual injection.
        self.edge_gate = nn.Sequential(
            nn.Conv2d(self.transport_dim + reduced_dim, reduced_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(reduced_dim, self.transport_dim, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv2d(self.transport_dim, dim, 1, 1, 0)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        _trunc_normal_(self.prototype_generator, std=0.02)

    def _directional_aggregation(self, reduced: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        responses = torch.stack(
            [branch(reduced) for branch in self.directional_bases], dim=1
        )  # B, M, C_r, H, W
        routes = torch.softmax(self.router(reduced), dim=1).unsqueeze(2)
        feat = torch.sum(routes * responses, dim=1)

        normalized_responses = [
            F.normalize(response, p=2, dim=1, eps=1e-6)
            for response in responses.unbind(dim=1)
        ]
        geometry = self.geometry_proj(torch.cat(normalized_responses, dim=1))
        return feat, geometry

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        n = h * w
        reduced = self.psi(x)
        feat, geometry = self._directional_aggregation(reduced)

        feat_flat = feat.flatten(2).transpose(1, 2)       # B, N, C_r
        geom_flat = geometry.flatten(2).transpose(1, 2)   # B, N, C_g

        # Soft prototype assignment S and spatially normalized assignment S_hat.
        assignment_logits = F.linear(feat_flat, self.prototype_generator)
        assignment = torch.softmax(assignment_logits / self.tau, dim=-1)
        spatial_norm = assignment.sum(dim=1, keepdim=True).clamp_min(1e-6)
        assignment_hat = assignment / spatial_norm

        # Compact structural prototypes P.
        prototypes = torch.bmm(assignment_hat.transpose(1, 2), feat_flat)

        # Prototype interaction kernel A_p.
        query = self.q_proj(prototypes)
        key = self.k_proj(prototypes)
        value = self.v_proj(prototypes)
        interaction = torch.softmax(
            torch.bmm(query, key.transpose(1, 2))
            / math.sqrt(self.transport_dim),
            dim=-1,
        )
        transported_rhs = torch.bmm(interaction, value)

        # Prototype-level geometry graph and Laplacian L_g.
        geom_proto = torch.bmm(assignment_hat.transpose(1, 2), geom_flat)
        geom_diff = geom_proto.unsqueeze(2) - geom_proto.unsqueeze(1)
        squared_distance = torch.sum(geom_diff * geom_diff, dim=-1)
        affinity = torch.exp(-squared_distance / max(self.sigma_g, 1e-6))
        degree = torch.sum(affinity, dim=-1)
        laplacian = torch.diag_embed(degree) - affinity

        eye = torch.eye(
            self.num_prototypes, device=x.device, dtype=laplacian.dtype
        ).unsqueeze(0)
        system = eye + self.lambda_g * laplacian

        # Solve in K x K prototype space; float32 solve is stable under AMP.
        solve_dtype = torch.float32 if system.dtype in (torch.float16, torch.bfloat16) else system.dtype
        refined_prototypes = torch.linalg.solve(
            system.to(solve_dtype),
            transported_rhs.to(solve_dtype),
        ).to(transported_rhs.dtype)

        # Pixel-to-prototype reassembly.
        location_query = self.location_proj(feat_flat)
        retrieval = torch.softmax(
            torch.bmm(location_query, refined_prototypes.transpose(1, 2))
            / math.sqrt(self.transport_dim),
            dim=-1,
        )
        reassembled = torch.bmm(retrieval, refined_prototypes)
        reassembled = reassembled.transpose(1, 2).reshape(
            b, self.transport_dim, h, w
        )

        # Edge-aware residual injection.
        edge_prior = reduced - F.avg_pool2d(reduced, 3, stride=1, padding=1)
        modulation = self.edge_gate(torch.cat([reassembled, edge_prior], dim=1))
        out = x + self.out_proj((1.0 + modulation) * reassembled)
        return out


class SDDRM(nn.Module):
    r"""Structure-Conditioned Detail Dictionary Refinement Module.

    It realizes:
        scale-aware structure/detail decomposition -> scene-adaptive detail
        dictionary -> detail reconstruction -> structure-conditioned
        calibration -> residual reinjection.

    Args:
        dim: Input/output channels.
        reduced_dim: Detail representation channels C_r.
        num_atoms: Number L of detail dictionary atoms.
        tau: Assignment temperature.
        offset_scale: Scale for scene-dependent dictionary offsets.
    """

    def __init__(
        self,
        dim: int = 64,
        reduced_dim: int = 32,
        num_atoms: int = 16,
        tau: float = 1.0,
        offset_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if reduced_dim <= 0 or num_atoms <= 0:
            raise ValueError("reduced_dim and num_atoms must be positive.")
        self.reduced_dim = reduced_dim
        self.num_atoms = num_atoms
        self.tau = float(tau)
        self.offset_scale = float(offset_scale)

        # eta(.) lightweight projection.
        self.eta = nn.Sequential(
            nn.Conv2d(dim, reduced_dim, 1, 1, 0),
            nn.GELU(),
        )

        # Multi-scale low-pass structural operators Lambda_s(.).
        self.smoothers = nn.ModuleList([
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=5, stride=1, padding=2),
            nn.AvgPool2d(kernel_size=7, stride=1, padding=3),
        ])
        self.scale_router = nn.Conv2d(reduced_dim, len(self.smoothers), 1, 1, 0)

        # Canonical detail dictionary D_0 and scene-conditioned adapter Theta(.).
        self.dictionary = nn.Parameter(torch.empty(num_atoms, reduced_dim))
        self.adapter = nn.Sequential(
            nn.Linear(reduced_dim, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, num_atoms * reduced_dim),
        )

        # zeta(.) and xi(.) for structural calibration and reinjection.
        self.calibration = nn.Sequential(
            nn.Conv2d(3 * reduced_dim, reduced_dim, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv2d(reduced_dim, dim, 1, 1, 0)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        _trunc_normal_(self.dictionary, std=0.02)

    def forward(self, h_in: torch.Tensor) -> torch.Tensor:
        b, _, h, w = h_in.shape
        reduced = self.eta(h_in)

        # Adaptive multi-scale structure-detail decomposition.
        scale_weights = torch.softmax(self.scale_router(reduced), dim=1).unsqueeze(2)
        low_pass = torch.stack([op(reduced) for op in self.smoothers], dim=1)
        structure = torch.sum(scale_weights * low_pass, dim=1)
        detail = reduced - structure

        # Scene-adaptive detail dictionary D_tilde.
        context = F.adaptive_avg_pool2d(structure, 1).flatten(1)
        offsets = self.adapter(context).view(b, self.num_atoms, self.reduced_dim)
        adapted_dictionary = self.dictionary.unsqueeze(0) + self.offset_scale * offsets

        # Detail assignment A_d and reconstructed detail R_hat.
        detail_flat = detail.flatten(2).transpose(1, 2)
        logits = torch.bmm(detail_flat, adapted_dictionary.transpose(1, 2))
        assignment = torch.softmax(logits / self.tau, dim=-1)
        reconstructed = torch.bmm(assignment, adapted_dictionary)
        reconstructed = reconstructed.transpose(1, 2).reshape(
            b, self.reduced_dim, h, w
        )

        # Structure-conditioned calibration.
        gate = self.calibration(
            torch.cat([reconstructed, structure, reconstructed * structure], dim=1)
        )
        out = h_in + self.out_proj(structure + (1.0 + gate) * reconstructed)
        return out


class GDCB(nn.Module):
    """Geometry-Detail Collaboration Block: GRPTM followed by SDDRM."""

    def __init__(
        self,
        dim: int,
        reduced_dim: int,
        num_prototypes: int,
        num_atoms: int,
        geom_dim: int,
        transport_dim: Optional[int],
        tau_p: float,
        tau_d: float,
        sigma_g: float,
        lambda_g: float,
        dict_offset_scale: float,
    ) -> None:
        super().__init__()
        self.grptm = GRPTM(
            dim=dim,
            reduced_dim=reduced_dim,
            num_prototypes=num_prototypes,
            geom_dim=geom_dim,
            transport_dim=transport_dim,
            tau=tau_p,
            sigma_g=sigma_g,
            lambda_g=lambda_g,
        )
        self.sddrm = SDDRM(
            dim=dim,
            reduced_dim=reduced_dim,
            num_atoms=num_atoms,
            tau=tau_d,
            offset_scale=dict_offset_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sddrm(self.grptm(x))


@ARCH_REGISTRY.register()
class GDCN(nn.Module, PyTorchModelHubMixin):
    r"""Geometry-Detail Collaborative Network for remote sensing image SR.

    The deprecated SMSR/Mamba arguments are retained where practical to make
    migration from an existing BasicSR YAML easier.  ``depths`` controls the
    number of stacked GDCBs through ``sum(depths)`` unless ``num_blocks`` is
    specified explicitly.

    Recommended starting configuration for x4:
        embed_dim=64, reduced_dim=32, num_blocks=6,
        num_prototypes=16, num_atoms=16, geom_dim=16.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 64,
        depths: Sequence[int] = (1, 1, 1, 1, 1, 1),
        num_blocks: Optional[int] = None,
        reduced_dim: int = 32,
        num_prototypes: int = 16,
        num_atoms: int = 16,
        geom_dim: int = 16,
        transport_dim: Optional[int] = None,
        tau_p: float = 1.0,
        tau_d: float = 1.0,
        sigma_g: float = 1.0,
        lambda_g: float = 0.1,
        dict_offset_scale: float = 0.1,
        upscale: int = 4,
        img_range: float = 1.0,
        upsampler: str = "pixelshuffledirect",
        resi_connection: str = "1conv",
        num_heads: Optional[Sequence[int]] = None,
        base_win_size: Optional[Sequence[int]] = None,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.0,
        value_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        hier_win_ratios: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        del img_size, patch_size, num_heads, base_win_size, mlp_ratio
        del drop_rate, value_drop_rate, drop_path_rate, norm_layer
        del ape, patch_norm, use_checkpoint, hier_win_ratios, kwargs

        num_out_ch = in_chans
        num_feat = embed_dim
        self.img_range = float(img_range)
        self.upscale = int(upscale)
        self.upsampler = upsampler
        self.num_blocks = int(num_blocks if num_blocks is not None else sum(depths))

        if in_chans == 3:
            mean = torch.tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1)
        else:
            mean = torch.zeros(1, in_chans, 1, 1)
        self.register_buffer("mean", mean, persistent=False)

        # 1. Shallow feature extraction H_s(.).
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # 2. Stacked Geometry-Detail Collaboration Blocks.
        self.body = nn.Sequential(*[
            GDCB(
                dim=embed_dim,
                reduced_dim=reduced_dim,
                num_prototypes=num_prototypes,
                num_atoms=num_atoms,
                geom_dim=geom_dim,
                transport_dim=transport_dim,
                tau_p=tau_p,
                tau_d=tau_d,
                sigma_g=sigma_g,
                lambda_g=lambda_g,
                dict_offset_scale=dict_offset_scale,
            )
            for _ in range(self.num_blocks)
        ])

        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            hidden = max(embed_dim // 4, 1)
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, hidden, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(hidden, hidden, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(hidden, embed_dim, 3, 1, 1),
            )
        else:
            raise ValueError("resi_connection must be '1conv' or '3conv'.")

        # 3. High-quality image reconstruction H_up(.).
        if self.upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)
        elif self.upsampler == "nearest+conv":
            if self.upscale != 4:
                raise ValueError("nearest+conv reconstruction currently supports x4 only.")
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        # Keep convolution layers under PyTorch's stable default initialization.
        # Linear projections follow the common restoration-network initialization.
        if isinstance(module, nn.Linear):
            _trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        x = (x - mean) * self.img_range

        shallow = self.conv_first(x)
        deep = self.conv_after_body(self.forward_features(shallow)) + shallow

        if self.upsampler == "pixelshuffle":
            out = self.conv_last(self.upsample(self.conv_before_upsample(deep)))
        elif self.upsampler == "pixelshuffledirect":
            out = self.upsample(deep)
        elif self.upsampler == "nearest+conv":
            feat = self.conv_before_upsample(deep)
            feat = self.lrelu(
                self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
            )
            feat = self.lrelu(
                self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
            )
            out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        else:
            out = x + self.conv_last(deep)

        out = out / self.img_range + mean
        if self.upsampler in {"pixelshuffle", "pixelshuffledirect", "nearest+conv"}:
            return out[:, :, : h * self.upscale, : w * self.upscale]
        return out[:, :, :h, :w]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = GDCN(
        upscale=4,
        embed_dim=64,
        num_blocks=6,
        reduced_dim=32,
        num_prototypes=16,
        num_atoms=16,
        geom_dim=16,
        upsampler="pixelshuffledirect",
    )
    model.eval()
    input_tensor = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Input shape:  {tuple(input_tensor.shape)}")
    print(f"Output shape: {tuple(output_tensor.shape)}")
    print(f"Trainable parameters: {params / 1e6:.3f} M")
