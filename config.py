
class ExperimentConfig:
    use_source_code=True
    use_difficulty=False
    model_name="distilbert-base-uncased"
    learning_rate=2e-5
    num_epochs=4
    dropout_rate=0.1
    batch_size=4
    data_path='./code_classification_dataset/'
    data_labels=['math', 'graphs', 'strings', 'number theory', 'trees',
    'geometry', 'games', 'probabilities']
    max_len_description=2000
    max_len_in_out_description=2000
    max_len_code=2000
    test_only=False
    test_dataset=None
    checkpoint_path='best_model.pth'
