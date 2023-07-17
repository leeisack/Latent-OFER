
def create_model(opt,model_fer=None):
    model = None
    print(opt.model)
    if opt.model == 'hybrid_net':
        from .hybrid import SelfAssembly
        model = SelfAssembly()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt,model_fer)
    
    print("model [%s] was created" % (model.name()))
    return model
