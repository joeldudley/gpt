from gpt.gpt import GPT
from tests.addition.constants import VOCAB_SIZE, MAX_TEST_SET_SIZE
from tests.addition.dataset import AdditionDataset
from tests.addition.evaluator import Evaluator
from tests.addition.samples import Samples
from trainer.trainer import Trainer


class Addition:
    def __init__(self, seed, num_digits):
        # a,b,a+b, and +1 due to potential carry overflow,
        # todo - joel - what does this mean?
        # but then also -1 because very last digit doesn't ever plug back
        # as there is no explicit <EOS> token to predict, it is implied
        max_seq_len = 3 * num_digits + 1 - 1
        self.model = GPT(VOCAB_SIZE, max_seq_len)

        self.train_dataset, self.test_dataset = self._get_datasets(num_digits, seed)
        self.evaluator = Evaluator(num_digits, self.train_dataset, self.test_dataset, self.model)

    def run(self, batch_end_callback, max_iterations):
        Trainer(self.model, self.train_dataset, batch_end_callback, max_iterations).run()

    def evaluate(self, trainer):
        return self.evaluator.evaluate(trainer)

    @staticmethod
    def _get_datasets(num_digits, seed):
        dataset_size = (10 ** num_digits) ** 2
        test_set_size = min(int(dataset_size * 0.2), MAX_TEST_SET_SIZE)  # 20% of the whole dataset, max. 500

        samples = Samples.get_all_samples(seed, num_digits, dataset_size)
        train_samples = samples[test_set_size:]
        test_samples = samples[:test_set_size]

        return AdditionDataset(train_samples), AdditionDataset(test_samples)
