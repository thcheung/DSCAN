from .DualModel import DualModel

def get_model(model_name, hidden_dim, classes, dropout, language):
    if model_name == 'dual':
        return DualModel(hidden_dim, classes,
                            dropout, language)