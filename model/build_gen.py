import model.TinyCNN as TinyCNN


def Generator(dataset='ENABL3S', sensor_num=0):
    return TinyCNN.Feature(dataset=dataset,
                           sensor_num=sensor_num)


def Classifier(dataset='ENABL3S'):
    return TinyCNN.Predictor(dataset=dataset)


def DomainClassifier():
    return TinyCNN.DomainPredictor()
