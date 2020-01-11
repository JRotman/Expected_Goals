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
        for play in all_plays:
            # SHOT event is a shot that was saved
            if play["result"]["eventTypeId"] == "SHOT":
                if "coordinates" in play:
                    if "x" in play["coordinates"] and "y" in play["coordinates"]:
                        x.append(play["coordinates"]["x"])
                        y.append(play["coordinates"]["y"])
                        result.append("SAVE")
            # GOAL event is a shot that resulted in a goal
            elif play["result"]["eventTypeId"] == "GOAL":
                if "emptyNet" in play["result"]:
                    # Do not want to include empty netters, since they cannot be saved by goalie
                    if play["result"]["emptyNet"] is False:
                        if "coordinates" in play:
                            if "x" in play["coordinates"] and "y" in play["coordinates"]:
                                x.append(play["coordinates"]["x"])
                                y.append(play["coordinates"]["y"])
                                result.append("GOAL")
            elif play["result"]["eventTypeId"] == "MISSED_SHOT":
                if "coordinates" in play:
                    if "x" in play["coordinates"] and "y" in play["coordinates"]:
                        x.append(play["coordinates"]["x"])
                        y.append(play["coordinates"]["y"])
                        result.append("MISS")
            elif play["result"]["eventTypeId"] == "BLOCKED_SHOT":
                if "coordinates" in play:
                    if "x" in play["coordinates"] and "y" in play["coordinates"]:
                        x.append(play["coordinates"]["x"])
                        y.append(play["coordinates"]["y"])
                        result.append("BLOCK")

        # print(json.dumps(sog, indent=4, separators=(',', ':')))
        # Interested in using the x and y coordinates to determine shot quality
        # Want consistent continuous variables
        # TODO: Explore using secondary type? (tip-in, snap shot, etc.)
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
    train_coord = data.loc[:39999, ['x', 'y']].to_numpy()
    train_labels = data.loc[:39999, 'result'].to_numpy()
    valid_coord = data.loc[40000:59999, ['x', 'y']].to_numpy()
    valid_labels = data.loc[40000:59999, 'result'].to_numpy()
    test_coord = data.loc[60000:, ['x', 'y']].to_numpy()
    test_labels = data.loc[60000:, 'result'].to_numpy()

    # Create layers
    # Assuming primarily linear relationships between distance from goal and chance to score
    # relu, helps to define the higher chance of scoring from the center of the offensive zone
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(20, activation='linear'))
    model.add(tf.keras.layers.Dense(16, activation='linear'))
    model.add(tf.keras.layers.Dense(8, activation='linear'))
    model.add(tf.keras.layers.Dense(8, activation='linear'))
    model.add(tf.keras.layers.Dense(4, activation='linear'))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Use binary crossentropy since we have two results/distributions
    # Might want to look at something else, isn't being particularly helpful
    # in narrowing down best results
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(x=train_coord, y=train_labels, epochs=20, batch_size=512,
                        shuffle=True, validation_data=(valid_coord, valid_labels), verbose=2)

    # Run tf evaluation on test data
    results = model.evaluate(x=test_coord, y=test_labels, verbose=1, batch_size=512)

    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))


    # Run personal evaluation on test data
    check = model.predict(test_coord, batch_size=512)
    test_df = pd.DataFrame(list(zip(check[:, 0], test_labels)), columns=["prediction", "result"])
    # save straight comparison between expected goals and goals
    test_df.to_csv("test_prediction.csv")
    print(test_df.groupby('result')['prediction'].mean())

    # print(check)# test_coord[0, 0], test_coord[0, 1], test_labels[0], check[0])
    # plot the predicted data as a 2dhistogram, weigh by xgoals, and normalize
    heatmap, xedges, yedges = np.histogram2d(test_coord[:, 0], test_coord[:, 1], normed=True,
                                             bins=50, weights=check[:, 0])
    extent = [0, 100, -42.5, 42.5]
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.title("Predicted Expected Goals")
    plt.savefig("Prediction.png")

    # plot the real data as a 2dhistogram, weigh by actual result, and normalize
    heatmap, xedges, yedges = np.histogram2d(test_coord[:, 0], test_coord[:, 1], normed=True,
                                             bins=50, weights=test_labels)
    extent = [0, 100, -42.5, 42.5]
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.title("Actual Goals")
    plt.savefig("Reality.png")

    # Print the range of expected goals
    # If this is too small, might just be predicting most shots are saved (which is fair)
    # If this is too large, might be overvaluing some shots
    print(min(check[:, 0]), " - ", max(check[:, 0]))

    return model


def compareGame(model):
    # Pull a single game from the 2019 season
    url = "https://statsapi.web.nhl.com/api/v1/game/2019020004/feed/live"
    r = requests.get(url)
    data = r.json()
    # print(json.dumps(data, indent=4, separators=(',', ':')))
    away = data["gameData"]["teams"]["away"]["id"]
    home = data["gameData"]["teams"]["home"]["id"]
    all_plays = data["liveData"]["plays"]["allPlays"]
    away_shots = []
    away_x = []
    away_y = []
    home_shots = []
    home_x = []
    home_y = []

    # Same as pull_data, but we will want to split between teams
    # NOTE: Could do this in the future to compare models between different teams
    for play in all_plays:
        # SHOT event is a shot that was saved
        if play["result"]["eventTypeId"] == "SHOT":
            if "coordinates" in play:
                if "x" in play["coordinates"] and "y" in play["coordinates"]:
                    if play["team"]["id"] == away:
                        away_x.append(play["coordinates"]["x"])
                        away_y.append(play["coordinates"]["y"])
                        away_shots.append(0)
                    else:
                        home_x.append(play["coordinates"]["x"])
                        home_y.append(play["coordinates"]["y"])
                        home_shots.append(0)

        # GOAL event is a shot that resulted in a goal
        elif play["result"]["eventTypeId"] == "GOAL":
            if "emptyNet" in play["result"]:
                # Do not want to include empty netters, since they cannot be saved by goalie
                if play["result"]["emptyNet"] is False:
                    if "coordinates" in play:
                        if "x" in play["coordinates"] and "y" in play["coordinates"]:
                            if play["team"]["id"] == away:
                                away_x.append(play["coordinates"]["x"])
                                away_y.append(play["coordinates"]["y"])
                                away_shots.append(1)
                            else:
                                home_x.append(play["coordinates"]["x"])
                                home_y.append(play["coordinates"]["y"])
                                home_shots.append(1)

    # Predict expected goals using our model
    home_coord = np.column_stack((home_x, home_y))
    home_compare = np.asarray(home_shots)
    home_check = model.predict(home_coord)
    away_coord = np.column_stack((away_x, away_y))
    away_compare = np.asarray(away_shots)
    away_check = model.predict(away_coord)

    # Total xGoals is the sum of all of the expectations for each shot
    print("Away team xGoals: ", np.sum(away_check[:, 0]))
    print("Home team xGoals: ", np.sum(home_check[:, 0]))
    print("Away team Goals: ", np.sum(away_compare))
    print("Home team Goals: ", np.sum(home_compare))

    # Create CSVs to compare predictions to realities again
    pd.DataFrame(list(zip(home_check, home_compare)),
                 columns=["xGoal", "result"]).to_csv("home_prediction.csv")
    pd.DataFrame(list(zip(away_check, away_compare)),
                 columns=["xGoal", "result"]).to_csv("away_prediction.csv")


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
    # xGoal_model = runNN(shot_df)
    # compareGame(xGoal_model)

    # NOTE: So far, the model is too inconsistent, changes each time it is run, and xGoals
    # are not always close to each other for the same 2019 game, can range from 1.5 - 13.5
    # Might need to implement some regression to mean? Based on average save percentage?
    # NOTE: Also could just try a Bayes' classifier

    # Look into pulling data into redshift
    # Logistic Regression on the distance from goal
    # Will need to calculate distance






"""
    url = "https://statsapi.web.nhl.com/api/v1/game/2018020018/feed/live"
    r = requests.get(url)
    data = r.json()

    all_plays = data["liveData"]["plays"]["allPlays"]
    print(json.dumps(all_plays, indent=4, separators=(',', ':')))
"""