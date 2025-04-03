import tensorflow as tf
from enum import Enum

class MappingType(Enum):
    Identity = 0
    Linear = 1
    Affine = 2

class ODESolver(Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2

class LTCCell(tf.keras.layers.Layer):
    def __init__(self, num_units, input_mapping=MappingType.Affine, solver=ODESolver.SemiImplicit, ode_solver_unfolds=6, activation='tanh', **kwargs):
        super(LTCCell, self).__init__(**kwargs)
        self.num_units = num_units
        self.ode_solver_unfolds = ode_solver_unfolds
        self.solver = solver
        self.input_mapping = input_mapping
        if isinstance(activation, str):
            self.activation = tf.keras.activations.get(activation)
        else:
            self.activation = activation
        self.build_kernel = False

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.num_units), initializer='uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.num_units, self.num_units), initializer='uniform', name='recurrent_kernel')
        self.bias = self.add_weight(shape=(self.num_units,), initializer='zeros', name='bias')
        self.build_kernel = True
        super().build(input_shape)

    @property
    def state_size(self):
        return self.num_units

    @property
    def output_size(self):
        return self.num_units

    def get_config(self):
        config = super(LTCCell, self).get_config()
        config.update({
            "num_units": self.num_units,
            "ode_solver_unfolds": self.ode_solver_unfolds,
            "solver": self.solver.value,
            "input_mapping": self.input_mapping.value,
            "activation": tf.keras.activations.serialize(self.activation)
        })
        return config

    def call(self, inputs, states):
        if not self.build_kernel:
            self.build(inputs.shape)
        prev_state = states[0]
        z = tf.matmul(inputs, self.kernel) + tf.matmul(prev_state, self.recurrent_kernel) + self.bias
        next_state = self.activation(z)
        return next_state, [next_state]
