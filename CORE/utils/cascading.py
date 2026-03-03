import torch


def cascading_predict(
    texture_logits: torch.Tensor,
    texture2_logits: torch.Tensor,
    t_live: float = 0.95,
    t_spoof: float = 0.95,
    f_spoof: float = 0.99,
) -> torch.Tensor:
    p_tex = torch.softmax(texture_logits, dim=1)
    p_freq = torch.softmax(texture2_logits, dim=1)
    device = texture_logits.device
    final = torch.zeros(texture_logits.size(0), dtype=torch.long, device=device)
    tex_spoof = p_tex[:, 1] > t_spoof
    tex_live = p_tex[:, 0] > t_live
    final[tex_spoof] = 1
    final[tex_live] = 0
    uncertain = ~(tex_spoof | tex_live)
    if uncertain.any():
        freq_spoof = p_freq[:, 1] > f_spoof
        freq_trigger = uncertain & freq_spoof
        final[freq_trigger] = 1
        fallback = uncertain & ~freq_spoof
        if fallback.any():
            fallback_pred = p_tex.argmax(dim=1)
            final[fallback] = fallback_pred[fallback]
    return final
