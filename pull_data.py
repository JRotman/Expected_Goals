import yaml
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pull_shots():
    x = []
    y = []
    result = []
    for i in range(1, 1272):
        url = "https://statsapi.web.nhl.com/api/v1/game/201802{}/feed/live".format(str(i).zfill(4))
        r = requests.get(url)
        data = r.json()

        all_plays = data["liveData"]["plays"]["allPlays"]
        sog = []
        for play in all_plays:
            if play["result"]["eventTypeId"] == "SHOT":
                sog.append(play)
            elif play["result"]["eventTypeId"] == "GOAL":
                if "emptyNet" in play["result"]:
                    if play["result"]["emptyNet"] is False:
                        sog.append(play)

        # print(json.dumps(sog, indent=4, separators=(',', ':')))
        for shot in sog:
            if "coordinates" in shot:
                if "x" in shot["coordinates"] and "y" in shot["coordinates"]:
                    x.append(shot["coordinates"]["x"])
                    y.append(shot["coordinates"]["y"])
                    result.append(shot["result"]["eventTypeId"])
        print(str(i).zfill(4))
        print(len(result))
    return pd.DataFrame(list(zip(x, y, result)), columns=['x', 'y', 'result'])


if __name__ == '__main__':
    # shot_df = pull_shots()
    # shot_df.to_csv("shot_data.csv", index=False)
    shot_df = pd.read_csv('shot_data.csv')
    sns.scatterplot(data=shot_df, x='x', y='y', hue='result')
    plt.show()






"""
    url = "https://statsapi.web.nhl.com/api/v1/game/2018020018/feed/live"
    r = requests.get(url)
    data = r.json()

    all_plays = data["liveData"]["plays"]["allPlays"]
    print(json.dumps(all_plays, indent=4, separators=(',', ':')))
"""