import numpy as np 

class NumpyDeque(object):

	def __init__(self, max_capacity):
		self.max_capacity = max_capacity
		self.data = None
		self.pointer = None

	def add(self, data):
		if self.data is None:
			self.data = data[-self.max_capacity:]

			if self.data.shape[0] == self.max_capacity:
				self.pointer = 0

		else:
			input_len = data.shape[0]

			if self.data.shape[0] + input_len <= self.max_capacity:
				self.data = np.vstack((self.data, data))

				if self.data.shape[0] == self.max_capacity:
					self.pointer = 0

			elif input_len >= self.max_capacity:
				self.data = data[-self.max_capacity:]
				self.pointer = 0

			else:
				if self.pointer is None:
					num_old_to_use = self.max_capacity - input_len
					self.data = np.vstack((data, self.data[-num_old_to_use:]))
					self.pointer = input_len

				else:
					pointer_end = (self.pointer + input_len) % self.max_capacity

					if pointer_end > self.pointer:
						self.data[self.pointer:pointer_end] = data

					else:
						num_at_end = self.max_capacity - self.pointer
						self.data[self.pointer:] = data[:num_at_end]
						self.data[:pointer_end] = data[num_at_end:]

					self.pointer = pointer_end

	def view(self):
		return self.data

	def size(self):
		return 0 if self.data is None else self.data.shape[0]