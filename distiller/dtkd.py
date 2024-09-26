import torch
import torch.nn as nn
import torch.nn.functional as F

class DTKD(nn.Module):
    def __init__(self, temperature=4.0):
        super(DTKD, self).__init__()
        self.temperature = temperature

    def forward(self, logits_student, logits_teacher):
        # DTKD Loss
        reference_temp = self.temperature
        logits_student_max, _ = logits_student.max(dim=1, keepdim=True)
        logits_teacher_max, _ = logits_teacher.max(dim=1, keepdim=True)
        logits_student_temp = 2 * logits_student_max / (logits_teacher_max + logits_student_max) * reference_temp
        logits_teacher_temp = 2 * logits_teacher_max / (logits_teacher_max + logits_student_max) * reference_temp
        
        ourskd = nn.KLDivLoss(reduction='none')(
            F.log_softmax(logits_student / logits_student_temp, dim=1),
            F.softmax(logits_teacher / logits_teacher_temp, dim=1)
        )
        loss_ourskd = (ourskd.sum(1, keepdim=True) * logits_teacher_temp * logits_student_temp).mean()
        
        return loss_ourskd