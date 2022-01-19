# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:26:42 2020

@author: Connor
"""

#
# Imports
#
import requests as reqs
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

_BASE ="https://collegefootballrisk.com/api"

plt.style.use("bmh")

class Territory:
    def __init__(self):
        self.name = None
        self.occupier = None
        self.winner = None
        self.teams = []
    
    def __repr__(self):
        if self.name and self.occupier:
            rep = f"""Territory<{self.name} owned by {self.occupier}>"""
        elif self.name:
            rep = f"""Territory<{self.name}>"""
        else:
            rep = "Territory<>"
        return rep

class Team:
    def __init__(self):
        self.name = None
        self.p_color = None
        self.s_color = None
        self.power = None
        self.chance = None
        self.players = None
        
    def __repr__(self):
        if self.name:
            rep = f"""Team<{self.name}>"""
        else:
            rep = "Team<>"
        return rep
    
def yline(loc, *args, ax=None, **kwargs):
    if ax is None:
        ylims = plt.ylim()
        plt.plot([loc, loc], ylims, *args, **kwargs)
        plt.ylim(ylims)
    else:
        ylims = ax.get_ylim()
        ax.plot([loc, loc], ylims, *args, **kwargs)
        ax.set_ylim(ylims)
        
def make_territory_list(day, season=2):
    """
    This simply does the api call for me for the day.
    """
    territory_req = reqs.get(_BASE+"/territories",
                         params={"season": season,
                                 "day": day})
    territories_list = territory_req.json()
    territory_list = []
    for terry in territories_list:
        tory = Territory()
        tory.name = terry["name"]
        territory_list.append(tory)
    
    return territory_list

def populate_territories(territory_list, day, season=2):
    for terry in territory_list:
        set_territory_data(terry, day, season=2)
        
    return territory_list

def set_territory_data(terry: Territory, day, season=2):
    """
    Idea is to have a large list of Territory objects which is populated via
    the /territory/turn api call.
    """
    territory_req = reqs.get(_BASE+"/territory/turn",
                             params={"territory": terry.name,
                                     "season": season,
                                     "day": day})
    territory_info = territory_req.json()
    terry.occupier = territory_info["occupier"]
    terry.winner = territory_info["winner"]
        
    for this_team in territory_info["teams"]:
        team = Team()
        team.name = this_team["team"]
        team.p_color = this_team["color"]
        team.s_color = this_team["secondaryColor"]
        team.power = this_team["power"]
        team.chance = this_team["chance"]
        team.players = this_team["players"]
        terry.teams.append(team)

def get_alive_teams(day, season=2, flag=False, num_runs=100000):
    leader_req = reqs.get(_BASE+"/stats/leaderboard",
                          params={"season": season,
                                  "day": day})
    leader_info = leader_req.json()
    alive_teams = {}
    for leader in leader_info:
        if flag:
            alive_teams[leader["name"]] = np.zeros((num_runs,))
        else:
            alive_teams[leader["name"]] = 0
    return alive_teams
    
def get_position(run_leaderboard, team_name):
    teams = [team[0] for team in run_leaderboard]
    vals = [team[1] for team in run_leaderboard]
    
    for ind, team in enumerate(teams):
        if team == team_name:
            ii = ind
            pos_delta = 0
            while ii > 0:
                if vals[ii-1] == vals[ii]:
                    pos_delta += 1
                else:
                    break
                ii -= 1
            break
    return ind+1-pos_delta
       
def get_turn_teams(day, season=50):
    leader_req = reqs.get(_BASE+"/stats/leaderboard",
                      params={"season": season,
                              "day": day})
    leader_info = leader_req.json()
    
    team_req = reqs.get(_BASE+"/teams")
    teams_info = team_req.json()
    team_list = []
    for leader in leader_info:
        for team in teams_info:
            if team["name"] == leader["name"]:
                this_team = Team()
                this_team.name = team["name"]
                this_team.p_color = team["colors"]["primary"]
                this_team.s_color = team["colors"]["secondary"]
                team_list.append(this_team)
                break
    return team_list

#%%
day = 50
season = 2
territory_list = make_territory_list(day)
territory_list = populate_territories(territory_list, day)
# So at this point, I have a list of each territory, with each team and 
# their odds
#%
position_dict = get_alive_teams(day, flag=True)
init_teams = get_alive_teams(day)
num_runs = 1
for run in range(num_runs):
    if run % 1000 == 0:
        print(f"Working on run: {run}")
    rand_vals = np.random.rand(len(territory_list))
    alive_teams = init_teams.copy()
    for ind, terry in enumerate(territory_list):
        odds = 0
        if terry.teams:
            for team in terry.teams:
                odds += team.chance
                if rand_vals[ind] <= odds:
                    alive_teams[team.name] += 1
                    break
        else:
            alive_teams[team.name] += 1
    run_leaderboard = sorted(alive_teams.items(), 
                             key=lambda x: x[1], 
                             reverse=True)
    for team in position_dict:
        position_dict[team][run] = get_position(run_leaderboard, team)    
    
                
#%%
leader_req = reqs.get(_BASE+"/stats/leaderboard",
                  params={"season": season,
                          "day": day+1})
leader_info = leader_req.json()
team_list = get_turn_teams(day, season=2)
for team in team_list:
    for leader in leader_info:
        if leader["name"] == team.name:
            break

    fig = plt.figure()
    ax = fig.gca()
    pos = position_dict[team.name]
    pos = pos[pos > 0]
    max_pos = max(pos)
    min_pos = min(pos)
    mu = np.mean(pos)
    sigma = np.std(pos)
    x_vals = np.arange(1, max_pos+1)
    y_vals = np.array([sum(pos == i) for i in range(1, int(max_pos+1))])*100 / len(pos)
    plt.bar(x_vals, 
            y_vals,
            0.9,
            align="center",
            color=team.p_color,
            edgecolor=team.s_color
            )
    if leader["rank"] > np.mean(pos):
        act_color = "#781b0e"
    else:
        act_color = "#3b8750"
    
    yline(leader["rank"], linestyle=(0,(2,2)), linewidth=2, color=act_color, label="Actual Rank")
    yline(mu, linestyle=(2,(1,1)), linewidth=1, color="#081840", label="Mean Rank")
    plt.title(f"Position Histogram for {team.name}")
    plt.xlabel("Leaderboard Position")
    plt.ylabel("Percent Odds for Turn 50")
    
    my_anno_text = f"""$\mu = {mu:2.3f}$
$3\sigma = {3*sigma:2.3f}$"""
    
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    if (mu) < (x_max-x_min)//2:
        # put both on right:
        ax.legend(loc="upper right")
        ax.text(0.85, 
                 0.08, 
                 my_anno_text, 
                 bbox={'facecolor': 'white', 'alpha': 0.7},
                 transform=ax.transAxes)
    # elif vals[0] > 5:
    #     # top
    #     ax.legend(loc="upper left")
    #     ax.text(0.72, 
    #              0.80, 
    #              my_anno_text, 
    #              bbox={'facecolor': 'white', 'alpha': 0.7},
    #              transform=ax.transAxes)
    else:
        # left
        ax.legend(loc="upper left")
        ax.text(0.03, 
                 0.10, 
                 my_anno_text, 
                 bbox={'facecolor': 'white', 'alpha': 0.7},
                 transform=ax.transAxes) 
    
    fig.savefig(f"./cfb_risk/turn_50/{team.name}_histogram", dpi=150)
    
