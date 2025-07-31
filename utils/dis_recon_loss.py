import torch
import torch.nn.functional as F


def cosine_distance(x, y):
    x_normalized = F.normalize(x, p=2, dim=-1)
    y_normalized = F.normalize(y, p=2, dim=-1)
    cos_sim = torch.matmul(x_normalized, y_normalized.transpose(1, 2))  # (B, J, J)
    return cos_sim


def centerize(dist_matrix):
    B, J, _ = dist_matrix.size()
    # calculate row and column mean values
    row_mean = dist_matrix.mean(dim=2, keepdim=True)  # (B, J, 1)
    col_mean = dist_matrix.mean(dim=1, keepdim=True)  # (B, 1, J)
    global_mean = dist_matrix.mean(dim=[1, 2], keepdim=True)  # (B, 1, 1)
    # ccenterized
    centerized_matrix = dist_matrix - row_mean - col_mean + global_mean
    return centerized_matrix


def share_loss(shared_features):
    mean_shared = shared_features.mean(dim=1, keepdim=True)  # (B, 1, C)
    cos_sim = F.cosine_similarity(shared_features, mean_shared, dim=-1)  # (B, J)
    cos_distance = 1 - cos_sim  # (B, J)
    loss = cos_distance.mean()
    return loss


def distenglement_loss(shared_features, expert_features):
    # separation for the specific experts
    similarity_matrix_tmp1 = cosine_distance(expert_features, expert_features)
    similarity_matrix_tmp1 = centerize(similarity_matrix_tmp1)
    mask = torch.eye(similarity_matrix_tmp1.size(1), device=similarity_matrix_tmp1.device)
    similarity_matrix1 = similarity_matrix_tmp1 * (1 - mask)

    # decorreation for the share-specific experts
    similarity_matrix_tmp2 = cosine_distance(shared_features, expert_features)
    similarity_matrix_tmp2 = centerize(similarity_matrix_tmp2)
    similarity_matrix2 = similarity_matrix_tmp2 * mask

    # sum for return
    return torch.abs(similarity_matrix1).sum() / (1 - mask).sum() + torch.abs(similarity_matrix2).sum() / mask.sum()


def cross_expert_loss(shared_outputs,expert_outputs,recon_outputs,expert_inputs):
    loss_share = share_loss(shared_outputs)
    loss_distenglement = distenglement_loss(shared_outputs, expert_outputs)
    loss_recon = F.mse_loss(expert_inputs, recon_outputs)
    return loss_share, loss_distenglement, loss_recon
