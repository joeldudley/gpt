def print_progress(addition, trainer):
    if trainer.iteration % 500 == 0:
        qty_correct_train, qty_correct_test = addition.evaluate(trainer)
        share_correct_train = 100 * qty_correct_train / len(addition.train_dataset)
        share_correct_test = 100 * qty_correct_test / len(addition.test_dataset)

        print()
        print("Iteration", trainer.iteration)
        print("Train score: %.2f%% correct (%d/%d)" % (
            share_correct_train, qty_correct_train, len(addition.train_dataset)))
        print("Test score: %.2f%% correct (%d/%d)" % (
            share_correct_test, qty_correct_test, len(addition.test_dataset)))

    elif trainer.iteration % 10 == 0:
        print('.', end='', flush=True)