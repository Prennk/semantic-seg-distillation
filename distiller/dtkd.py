import torch
import torch.nn as nn
import torch.nn.functional as F

class DTKD(nn.Module):
    def __init__(self, temperature=4.0, alpha=1.0, beta=1.0, warmup=1, ce_loss_weight=1.0):
        super(DTKD, self).__init__()
        self.ce_loss_weight = ce_loss_weight
        self.alpha = alpha
        self.beta = beta
        self.warmup = warmup
        self.temperature = temperature

    def compute_loss(self, logits_student, logits_teacher, target, epoch):
        # DTKD Loss
        reference_temp = self.temperature
        logits_student_max, _ = logits_student.max(dim=1, keepdim=True)
        logits_teacher_max, _ = logits_teacher.max(dim=1, keepdim=True)
        
        # Compute temperature for student and teacher
        logits_student_temp = 2 * logits_student_max / (logits_teacher_max + logits_student_max) * reference_temp
        logits_teacher_temp = 2 * logits_teacher_max / (logits_teacher_max + logits_student_max) * reference_temp
        
        # Compute KL Divergence for our SKD
        ourskd = nn.KLDivLoss(reduction='none')(
            F.log_softmax(logits_student / logits_student_temp, dim=1),
            F.softmax(logits_teacher / logits_teacher_temp, dim=1)
        )
        loss_ourskd = (ourskd.sum(1, keepdim=True) * logits_teacher_temp * logits_student_temp).mean()
        # print(f"logits_student: {logits_student}")
        # print()
        # print(f"logits_student_max: {logits_student_max}")
        # print()
        # print(f"loss_ourskd: {loss_ourskd}")

        # Vanilla KD Loss
        vanilla_temp = self.temperature
        kd = nn.KLDivLoss(reduction='none')(
            F.log_softmax(logits_student / vanilla_temp, dim=1),
            F.softmax(logits_teacher / vanilla_temp, dim=1)
        )
        loss_kd = (kd.sum(1, keepdim=True) * vanilla_temp ** 2).mean()

        # CrossEntropy Loss
        loss_ce = nn.CrossEntropyLoss()(logits_student, target)

        # Total DTKD Loss
        loss_dtkd = min(epoch / self.warmup, 1.0) * (self.alpha * loss_ourskd + self.beta * loss_kd) + self.ce_loss_weight * loss_ce
        
        return loss_dtkd, self.ce_loss_weight * loss_ce, min(epoch / self.warmup, 1.0) * (self.alpha * loss_ourskd + self.beta * loss_kd)