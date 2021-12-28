#import tensorboard as tb
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np


SIZE_GUIDANCE = {'tensors': 100}
LEGEND = { 'epoch_bpp': 'bpp avr', 'epoch_bpp_min': 'bpp min', 'epoch_bpp_max': 'bpp max' }


def load_logs(path):
	event_acc = event_accumulator.EventAccumulator(path, SIZE_GUIDANCE)
	event_acc.Reload()
	return {
		k:np.asarray([(v.step, tf.make_ndarray(v.tensor_proto)) for v in event_acc.Tensors(k)]).T for k in LEGEND.keys()
	}

logs = load_logs('logs/r3_0.5M_uids/test/events.out.tfevents.1639981648.329299-dbxtree2-4-gz6n7.38.724.v2')
#logs = tb.data.experimental.ExperimentFromDev('logs/r3_0.5M_uids/test/events.out.tfevents.1639981648.329299-dbxtree2-4-gz6n7.38.724.v2')
#print(logs.__dict__)
#print(logs.get_tensors(pivot=True))

plt.title('UIDs')
for tag, vals in logs.items():
	plt.plot(vals[0], vals[1], label=LEGEND[tag])
plt.show()