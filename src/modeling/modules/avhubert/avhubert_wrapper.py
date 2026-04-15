import logging
import src.modeling.modules.avhubert
from fairseq import checkpoint_utils

logger = logging.getLogger(__name__)

def load_avhubert_model(checkpoint_path, modalities, use_cuda=False):
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
    model = models[0]
    
    if use_cuda:
        model.cuda()

    if hasattr(model, 'decoder'):
        model = model.encoder.w2v_model
        logger.info("Load AV-HuBERT fine-tuned checkpoint model.")
    else:
        logger.info("Load AV-HuBERT pre-trained w/o finetuned checkpoint model.")
    
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model
