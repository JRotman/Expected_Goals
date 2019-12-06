import json
import requests
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pull_shots():
    x = []
    y = []
    result = []
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

        print(json.dumps(sog, indent=4, separators=(',', ':')))
        # Interested in using the x and y coordinates to determine shot quality
        # Want consistent continuous variables
        # TODO: Explore using secondary type? (tip-in, snap shot, etc.
        for shot in sog:
            if "coordinates" in shot:
                if "x" in shot["coordinates"] and "y" in shot["coordinates"]:
                    x.append(shot["coordinates"]["x"])
                    y.append(shot["coordinates"]["y"])
                    result.append(shot["result"]["eventTypeId"])
        print(str(i).zfill(4))
        print(len(result))
        # Return a data frame
    return pd.DataFrame(list(zip(x, y, result)), columns=['x', 'y', 'result'])


if __name__ == '__main__':
    # Check if the data has been pulled already
    if os.path.exists("shot_data.csv"):
        shot_df = pd.read_csv('shot_data.csv')
    # If not pull it and save it to a CSV
    else:
        shot_df = pull_shots()
        shot_df.to_csv("shot_data.csv", index=False)
    
    sns.scatterplot(data=shot_df, x='x', y='y', hue='result')
    plt.show()






"""
    url = "https://statsapi.web.nhl.com/api/v1/game/2018020018/feed/live"
    r = requests.get(url)
    data = r.json()

    all_plays = data["liveData"]["plays"]["allPlays"]
    print(json.dumps(all_plays, indent=4, separators=(',', ':')))
"""