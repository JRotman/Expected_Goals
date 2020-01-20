import json
import requests
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.linear_model import LogisticRegression


def pull_shots(year):
    x = []
    y = []
    result = []
    games = 0
    games_parsed = 0
    if year < 2017:
        games = 1231
    else:
        games = 1272
    # Iterate over all 1271 games from the NHL season
    for i in range(1, games):
        # Must use 4 digit code for each game, starting from 0001
        url = "https://statsapi.web.nhl.com/api/v1/game/{}02{}/feed/live".format(str(year).zfill(4),
                                                                                 str(i).zfill(4))
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


def logit_corsi(corsi_df):
    # In Hockey Corsi is all shots taken
    corsi_df.loc[:, 'goal'] = np.where(corsi_df.loc[:, 'result'] == "GOAL", 1, 0)
    # Constants based on goal being at (89,0)
    # Absolute value on X to fold one side of the rink on the other
    corsi_df.loc[:, 'distance'] = np.sqrt(np.power(np.abs(corsi_df.loc[:, 'x']) - 89, 2) +
                                          np.power(corsi_df.loc[:, 'y'], 2))
    # Model based on linear relationship between distance from goal and likelihood of goal
    corsi_model = LogisticRegression(penalty='l2', solver='liblinear').fit(
        np.asarray(corsi_df['distance']).reshape(-1, 1), np.asarray(corsi_df['goal']))
    return corsi_model


def logit_fenwick(shot_df):
    # In Hockey Fenwick is all shots besides those that are blocked
    fenwick_df = pd.DataFrame(shot_df.loc[shot_df.loc[:, 'result'] != "BLOCK", :])
    fenwick_df.loc[:, 'goal'] = np.where(fenwick_df.loc[:, 'result'] == "GOAL", 1, 0)
    # Constants based on goal being at (89,0)
    # Absolute value on X to fold one side of the rink on the other
    fenwick_df.loc[:, 'distance'] = np.sqrt(np.power(np.abs(fenwick_df.loc[:, 'x']) - 89, 2) +
                                            np.power(fenwick_df.loc[:, 'y'], 2))
    fenwick_model = LogisticRegression(penalty='l2', solver='liblinear').fit(
        np.asarray(fenwick_df['distance']).reshape(-1, 1), np.asarray(fenwick_df['goal']))
    return fenwick_model


def logit_sog(shot_df):
    # In Hockey SoG is all shots that were goals, or saved by the goalie
    sog_df = pd.DataFrame(shot_df.loc[(shot_df.loc[:, 'result'] == "SAVE") |
                                      (shot_df.loc[:, 'result'] == "GOAL"), :])
    sog_df.loc[:, 'goal'] = np.where(sog_df.loc[:, 'result'] == "GOAL", 1, 0)
    # Constants based on goal being at (89,0)
    # Absolute value on X to fold one side of the rink on the other
    sog_df.loc[:, 'distance'] = np.sqrt(np.power(np.abs(sog_df.loc[:, 'x']) - 89, 2) +
                                        np.power(sog_df.loc[:, 'y'], 2))
    sog_model = LogisticRegression(penalty='l2', solver='liblinear').fit(
        np.asarray(sog_df['distance']).reshape(-1, 1), np.asarray(sog_df['goal']))
    return sog_model


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
    away_dist = []
    home_shots = []
    home_dist = []

    # Same as pull_data, but we will want to split between teams
    # NOTE: Could do this in the future to compare models between different teams
    for play in all_plays:
        # SHOT event is a shot that was saved
        if play["result"]["eventTypeId"] == "SHOT" or play["result"]["eventTypeId"] == "MISSED SHOT"\
                or play["result"]["eventTypeId"] == "BLOCKED SHOT":
            if "coordinates" in play:
                if "x" in play["coordinates"] and "y" in play["coordinates"]:
                    if play["team"]["id"] == away:
                        away_dist.append(np.sqrt(np.power(np.abs(play["coordinates"]["x"]) - 89, 2) +
                                                 np.power(play["coordinates"]["y"], 2)))
                        away_shots.append(0)
                    else:
                        home_dist.append(np.sqrt(np.power(np.abs(play["coordinates"]["x"]) - 89, 2) +
                                                 np.power(play["coordinates"]["y"], 2)))
                        home_shots.append(0)

        # GOAL event is a shot that resulted in a goal
        elif play["result"]["eventTypeId"] == "GOAL":
            if "emptyNet" in play["result"]:
                # Do not want to include empty netters, since they cannot be saved by goalie
                if play["result"]["emptyNet"] is False:
                    if "coordinates" in play:
                        if "x" in play["coordinates"] and "y" in play["coordinates"]:
                            if play["team"]["id"] == away:
                                away_dist.append(np.sqrt(np.power(np.abs(play["coordinates"]["x"]) - 89, 2) +
                                                         np.power(play["coordinates"]["y"], 2)))
                                away_shots.append(1)
                            else:
                                home_dist.append(np.sqrt(np.power(np.abs(play["coordinates"]["x"]) - 89, 2) +
                                                         np.power(play["coordinates"]["y"], 2)))
                                home_shots.append(1)

    # Predict expected goals using our model
    home_coord = np.asarray(home_dist).reshape(-1, 1)
    home_compare = np.asarray(home_shots)
    home_check = model.predict_proba(home_coord)
    away_coord = np.asarray(away_dist).reshape(-1, 1)
    away_compare = np.asarray(away_shots)
    away_check = model.predict_proba(away_coord)

    # Total xGoals is the sum of all of the expectations for each shot
    print("Away team xGoals: ", np.sum(away_check[:, 1]))
    print("Home team xGoals: ", np.sum(home_check[:, 1]))
    print("Away team Goals: ", np.sum(away_compare))
    print("Home team Goals: ", np.sum(home_compare))

    # Create CSVs to compare predictions to realities again
    pd.DataFrame(list(zip(home_check, home_compare)),
                 columns=["xGoal", "result"]).to_csv("home_prediction.csv")
    pd.DataFrame(list(zip(away_check, away_compare)),
                 columns=["xGoal", "result"]).to_csv("away_prediction.csv")


def test_model(model, test_data, model_name):
    # Run personal evaluation on test data
    test_data.loc[:, 'goal'] = np.where(test_data.loc[:, 'result'] == "GOAL", 1, 0)
    test_data.loc[:, 'distance'] = np.sqrt(np.power(np.abs(test_data.loc[:, 'x']) - 89, 2) +
                                            np.power(test_data.loc[:, 'y'], 2))
    check = model.predict_proba(np.asarray(test_data.loc[:, 'distance']).reshape(-1, 1))
    test_df = pd.DataFrame(list(zip(check[:, 1], test_data.loc[:, 'goal'])), columns=["prediction", "result"])
    # save straight comparison between expected goals and goals
    test_df.to_csv("{}_prediction.csv".format(model_name))
    # print(test_df.groupby('result')['prediction'].mean())

    # print(check)# test_coord[0, 0], test_coord[0, 1], test_labels[0], check[0])
    # plot the predicted data as a 2dhistogram, weigh by xgoals, and normalize
    heatmap, xedges, yedges = np.histogram2d(test_data.loc[:, 'x'], test_data.loc[:, 'y'], normed=True,
                                             bins=50, weights=check[:, 1])
    extent = [0, 100, -42.5, 42.5]
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.title("{} Predicted Expected Goals".format(model_name))
    plt.savefig("Prediction_{}.png".format(model_name))

    # plot the real data as a 2dhistogram, weigh by actual result, and normalize
    heatmap, xedges, yedges = np.histogram2d(test_data.loc[:, 'x'], test_data.loc[:, 'y'], normed=True,
                                             bins=50, weights=test_data.loc[:, 'goal'])
    extent = [0, 100, -42.5, 42.5]
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.title("Actual Goals")
    plt.savefig("Reality.png")

    xG = np.sum(test_df.loc[:, 'prediction'])
    G = np.sum(test_df.loc[:, 'result'])
    print(model_name, " xG - G = ", xG - G)


if __name__ == '__main__':
    # Check if the training data has been pulled already
    if os.path.exists("train_data.csv"):
        print("Training data located, using saved CSV")
        train_df = pd.read_csv('train_data.csv')
    # If not pull it and save it to a CSV
    else:
        print("Training data not found, now pulling 3 seasons (2015-2017) from NHL API")
        year1 = pull_shots(2015)
        year2 = pull_shots(2016)
        year3 = pull_shots(2017)
        train_df = pd.concat([year1, year2, year3])
        train_df.to_csv("train_data.csv", index=False)

    if os.path.exists("test_data.csv"):
        print("Testing data located, using saved CSV")
        test_df = pd.read_csv('test_data.csv')
    # If not pull it and save it to a CSV
    else:
        print("Testing data not found, now pulling one season (2018) from NHL API")
        test_df = pull_shots(2018)
        test_df.to_csv("test_data.csv", index=False)

    corsi_xg_model = logit_corsi(train_df)
    fenwick_xg_model = logit_fenwick(train_df)
    sog_xg_model = logit_sog(train_df)
    test_model(corsi_xg_model, test_df, "Corsi")
    test_model(fenwick_xg_model, test_df, "Fenwick")
    test_model(sog_xg_model, test_df, "SoG")

    compareGame(corsi_xg_model)
    compareGame(fenwick_xg_model)
    compareGame(sog_xg_model)

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