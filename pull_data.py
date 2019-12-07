import json
import requests
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


def pull_shots():
    x = []
    y = []
    result = []
    games_parsed = 0
    # Iterate over all 1271 games from the NHL season
    for i in range(1, 1272):
        # Must use 4 digit code for each game, starting from 0001
        url = "https://statsapi.web.nhl.com/api/v1/game/201802{}/feed/live".format(str(i).zfill(4))
        r = requests.get(url)
        data = r.json()

        # Ignore data on teams and players, just interested in events during game
        all_plays = data["liveData"]["plays"]["allPlays"]
        sog = []
        for play in all_plays:
            # SHOT event is a shot that was saved
            if play["result"]["eventTypeId"] == "SHOT":
                sog.append(play)
            # GOAL event is a shot that resulted in a goal
            elif play["result"]["eventTypeId"] == "GOAL":
                if "emptyNet" in play["result"]:
                    # Do not want to include empty netters, since they cannot be saved by goalie
                    if play["result"]["emptyNet"] is False:
                        sog.append(play)

        # print(json.dumps(sog, indent=4, separators=(',', ':')))
        # Interested in using the x and y coordinates to determine shot quality
        # Want consistent continuous variables
        # TODO: Explore using secondary type? (tip-in, snap shot, etc.)
        for shot in sog:
            if "coordinates" in shot:
                if "x" in shot["coordinates"] and "y" in shot["coordinates"]:
                    x.append(shot["coordinates"]["x"])
                    y.append(shot["coordinates"]["y"])
                    result.append(1 if shot["result"]["eventTypeId"] == 'GOAL' else 0)
        # print(str(i).zfill(4))
        # print(len(result))
        # Return a data frame
        games_parsed += 1
        if games_parsed % 25 == 0:
            print('{0:.1f}% completed'.format(100*(games_parsed / 1271)))
    return pd.DataFrame(list(zip(x, y, result)), columns=['x', 'y', 'result'])


def runNN(data):
    # Shuffle data to mix up games (prevent overfitting to beginning of season)
    data = data.sample(frac=1).reset_index(drop=True)

    # Increase power by treating each half of the ice as the same
    data.loc[:, 'x'] = data.loc[:, 'x'].abs()

    # Split up data into training, validation, and testing sets
    train_coord = data.loc[:44999, ['x', 'y']].to_numpy()
    train_labels = data.loc[:44999, 'result'].to_numpy()
    valid_coord = data.loc[45000:59999, ['x', 'y']].to_numpy()
    valid_labels = data.loc[45000:59999, 'result'].to_numpy()
    test_coord = data.loc[60000:, ['x', 'y']].to_numpy()
    test_labels = data.loc[60000:, 'result'].to_numpy()

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(2, activation='elu'))
    model.add(tf.keras.layers.Dense(4, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(x=train_coord, y=train_labels, epochs=20, batch_size=512,
                        shuffle=True, validation_data=(valid_coord, valid_labels), verbose=2)

    results = model.evaluate(x=test_coord, y=test_labels, verbose=1, batch_size=512)

    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))

    check = model.predict(test_coord, batch_size=512)
    # print(check)# test_coord[0, 0], test_coord[0, 1], test_labels[0], check[0])
    heatmap, xedges, yedges = np.histogram2d(test_coord[:, 0], test_coord[:, 1],
                                            bins=50, weights=check[:, 0], density=True)
    extent = [0, 100, -42.5, 42.5]
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.title("Predicted Expected Goals")
    plt.savefig("Prediction.png")

    heatmap, xedges, yedges = np.histogram2d(test_coord[:, 0], test_coord[:, 1],
                                             bins=50, weights=test_labels, density=True)
    extent = [0, 100, -42.5, 42.5]
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.title("Actual Goals")
    plt.savefig("Reality.png")

    return model


if __name__ == '__main__':
    # Check if the data has been pulled already
    if os.path.exists("shot_data.csv"):
        print("Shot data located, using saved CSV")
        shot_df = pd.read_csv('shot_data.csv')
    # If not pull it and save it to a CSV
    else:
        print("Data not found, now pulling from NHL API")
        shot_df = pull_shots()
        shot_df.to_csv("shot_data.csv", index=False)

    # print(len(shot_df.index))
    # In total 79822 points of data
    xGoal_model = runNN(shot_df)






"""
    url = "https://statsapi.web.nhl.com/api/v1/game/2018020018/feed/live"
    r = requests.get(url)
    data = r.json()

    all_plays = data["liveData"]["plays"]["allPlays"]
    print(json.dumps(all_plays, indent=4, separators=(',', ':')))
"""