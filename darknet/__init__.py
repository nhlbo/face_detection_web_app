import logging
from darknet import model

logger = logging.getLogger(__name__)
logging.info('loading model')
model.load_model()