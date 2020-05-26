# dx to be time series data of spread
threshold = 0.25
length = 5

def metric1(dx):
    label = []
    # run the entire time series
    for t in range(0, len(dx)-length):
        profit = False

        # check profitablity i periods ahead
        for i in range(1, length+1):

            if dx[t+i] <= dx[t]*threshold:
                profit = True
                break

        # profitable, label = 1 , otherwise, label = -1
        if profit == True:
            label.append(1)
        else:
            label.append(-1)

    return label


def metric2(dx):
    label = []

    for t in range(0, len(dx)-1):

        if dx[t] >= dx[t+1]:
            label.append(1)
        else:
            label.append(-1)

    return label
