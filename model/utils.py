# -- coding: utf-8 --
import numpy as np

def construct_feed_dict(features, labels, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    return feed_dict

def accuracy(label, predict):
    '''
    :param label: represents the observed value
    :param predict: represents the predicted value
    :param epoch:
    :param steps:
    :return:
    '''
    error = label - predict
    average_error = np.mean(np.fabs(error.astype(float)))
    print("MAE is : %.6f" % (average_error))

    rmse_error = np.sqrt(np.mean(np.square(label - predict)))
    print("RMSE is : %.6f" % (rmse_error))

    cor = np.mean(np.multiply((label - np.mean(label)),
                              (predict - np.mean(predict)))) / (np.std(predict) * np.std(label))
    print('R is: %.6f' % (cor))

    # mask = label != 0
    # mape =np.mean(np.fabs((label[mask] - predict[mask]) / label[mask]))*100.0
    # mape=np.mean(np.fabs((label - predict) / label)) * 100.0
    # print('mape is: %.6f %' % (mape))
    sse = np.sum((label - predict) ** 2)
    sst = np.sum((label - np.mean(label)) ** 2)
    R2 = 1 - sse / sst  # r2_score(y_actual, y_predicted, multioutput='raw_values')
    print('R^$2$ is: %.6f' % (R2))

    return average_error, rmse_error, cor, R2

def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        # mae = np.nan_to_num(mae * mask)
        # wape = np.divide(np.sum(mae), np.sum(label))
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
        cor = np.mean(np.multiply((label - np.mean(label)),
                                  (pred - np.mean(pred)))) / (np.std(pred) * np.std(label))
        sse = np.sum((label - pred) ** 2)
        sst = np.sum((label - np.mean(label)) ** 2)
        r2 = 1 - sse / sst  # r2_score(y_actual, y_predicted, multioutput='raw_values')
        print('mae is : %.6f'%mae)
        print('rmse is : %.6f'%rmse)
        print('mape is : %.6f'%mape)
        print('r is : %.6f'%cor)
        print('r$^2$ is : %.6f'%r2)
    return mae, rmse, mape, cor, r2

