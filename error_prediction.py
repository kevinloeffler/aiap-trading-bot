# ToDo: required variables: predictions, real_stock_price and training history object

def calculate_error(history, predictions, real_stock_price):
    # Calculate Error
    positive_errors = []
    negative_errors = []
    for i in range(len(predictions)):
        error = predictions[i] - real_stock_price[i]
        if error > 0:
            positive_errors.append(error)
        elif error < 0:
            negative_errors.append(error)

    error_pos, error_neg = 0, 0

    for error in positive_errors:
        error_pos += error

    for error in negative_errors:
        error_neg += error

    error_pos = error_pos / len(positive_errors)
    error_neg = error_neg / len(negative_errors)

    print(f'Positive Error = {error_pos}')  # The average error in the positive direction
    print(f'Negative Error = {error_neg}')  # The average error in the negative direction
    print(f'Training loss = {history.history["loss"][-1]}')  # The training loss on the last epoch
    return error_pos, error_neg
