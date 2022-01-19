# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 23:02:12 2020

@author: Connor

This file will be my CFB risk modules.
"""

#
# Imports
#
import requests as reqs
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import erf

_BASE ="https://collegefootballrisk.com/api"
_SEASON = 1
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
        
    def __repr__(self):
        if self.name:
            rep = f"""Team<{self.name}>"""
        else:
            rep = "Team<>"
        return rep

def make_territory_list(day, season=_SEASON):
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

def populate_territories(territory_list):
    for terry in territory_list:
        set_territory_data(terry)
        
    return territory_list

def set_territory_data(terry: Territory, day, season=_SEASON):
    """
    Idea is to have a large list of Territory objects which is populated via
    the /territory/turn api call.
    """
    territory_req = reqs.get(_BASE+"/territory/turn",
                          params={"season": season,
                                  "day": day,
                                  "team": terry.name})
    territory_info = territory_req.json()
    terry.occupier = territory_info["occupier"]
    terry.winner = territory_info["winner"]
    
    for tory in territory_req:
        
        for this_team in tory["teams"]:
            team = Team()
            team.name = this_team["team"]
            team.p_color = this_team["color"]
            team.s_color = this_team["secondaryColor"]
            team.power = this_team["power"]


def yline(loc, *args, ax=None, **kwargs):
    if ax is None:
        ylims = plt.ylim()
        plt.plot([loc, loc], ylims, *args, **kwargs)
        plt.ylim(ylims)
    else:
        ylims = ax.get_ylim()
        ax.plot([loc, loc], ylims, *args, **kwargs)
        ax.set_ylim(ylims)

def create_expected_value_hist(
        team_name,
        day,
        prev_num_terry,
        num_runs=100000,
        season=_SEASON,
        axis=None,
        save_dir=None
        ):
    """
    ``create_expected_value_hist``, as the name suggests, creates an expected 
    value histogram for a given team and day from the data in the CFB_RISK api.  
    
    if ax = None, plt.gca() is used.
    """
    try:
        team_odds_req = reqs.get(_BASE+"/team/odds",
                             params={"season": season,
                                     "day": day,
                                     "team": team_name})
        team_odds_info = team_odds_req.json()
        
        teams_req = reqs.get(_BASE+"/teams")
        team_info = teams_req.json()
    
        p_color = None
        for team in team_info:
            if team["name"] == team_name:
                p_color = team["colors"]["primary"]
                s_color = team["colors"]["secondary"]
                break
        
        if p_color is None:
            raise ValueError(f"Invalid team_name = {team_name}")
            
        p_color = tuple(float(val)/255 if ii < 3 else float(val) for ii, val in enumerate(p_color[5:-1].split(",")))
        s_color = tuple(float(val)/255 if ii < 3 else float(val) for ii, val in enumerate(s_color[5:-1].split(",")))
        
        if p_color[0:3] == (1, 1, 1):
            p_color = (0, 0, 0, p_color[3])
        if s_color[0:3] == (1, 1, 1):
            s_color = (0, 0, 0, s_color[3])
    
        num_territories = len(team_odds_info)
        # start with a vector of ones (the "empty territories have a chance of 1)
        odds = np.ones((num_territories,))
        
        # for each territoy, exluding "all", compute exact odds
        odds = [tory["teamPower"]/tory["territoryPower"]  # put the stats, else 1
                    if tory["territoryPower"]>0 else 1 # if denom != 0
                    for tory in team_odds_info] # for tory in odds_info
        
        # This calculates the PDF
        vals = 1
        for k in odds:
            vals = np.convolve(vals, [1-k, k])
        
        # axis handling
        if axis is None:
            fig = plt.figure()
            _ax = plt.gca()
        else:
            _ax = axis
        
        # set up plot values
        act = sum([1 if tory["winner"] == team_name else 0 for tory in team_odds_info])
        exp = sum(odds)
        # Gets the Expected Value numerically to validate expected Odds
        mu = np.sum(vals*np.arange(len(vals)))
        # Gets the Sigma numerically to validate variance
        sigma = np.sqrt(sum(vals*(np.arange(len(vals)) - mu)**2))
        dsigma = (act-mu) / sigma
        # draw_percentage = stats.norm.pdf(dsigma)*100
        
        if dsigma < 0:
            act_color = "#781b0e"
        else:
            act_color = "#3b8750"
        
        x = np.linspace(0, num_territories, 5000)
        y = (1 / (np.sqrt(2 * np.pi * np.power(sigma, 2)))) * \
            (np.power(np.e, -(np.power((x - mu), 2) / (2 * np.power(sigma, 2)))))
        cdf = 0.5 *  (1 + erf((act-exp)/(np.sqrt(2)*sigma)))
        _ax.plot(x,y*100, linestyle="-", linewidth=0.5, color="#54585A", label="$X$ ~ $N(\mu, \sigma)$")
        _ax.bar(np.arange(num_territories+1), vals*100, 0.9, align="center", color=p_color, edgecolor=s_color)
        yline(exp, ax=_ax, linestyle=(0,(2,2)), linewidth=2, color="#081840", label="Expected Value")
        yline(act, ax=_ax, linestyle=(0,(2,2)), linewidth=2, color=act_color, label="Actual Territories")
        yline(prev_num_terry, ax=_ax, linestyle=(0,(1,1)), linewidth=2, color="#ffb521", label="Prev Num. Territories")
        dT = act - prev_num_terry
        _ax.set_title(f"Number of Territories Histogram: {team_name}\n$Expected: {exp:2.2f}$, $Actual: {act}$, $\Delta Territories = {dT}$")
        _ax.set_xlabel("Number of Territories Won")
        _ax.set_ylabel("Percent Chance to Win N Territories (%)")
        my_anno_text = f"""$\mu = {mu:2.3f}$
    $3\sigma = {3*sigma:2.3f}$
    $\Delta\sigma = {dsigma:2.3f}$
    $P(Draw) = {100*vals[act]:2.3f}\%$"""
        
        x_min, x_max = _ax.get_xlim()
        y_min, y_max = _ax.get_ylim()
        if (mu) < (x_max-x_min)//2:
            # put both on right:
            _ax.legend(loc="upper right")
            _ax.text(0.72, 
                     0.08, 
                     my_anno_text, 
                     bbox={'facecolor': 'white', 'alpha': 0.7},
                     transform=_ax.transAxes)
        elif vals[0] > 5:
            # top
            _ax.legend(loc="upper left")
            _ax.text(0.72, 
                     0.80, 
                     my_anno_text, 
                     bbox={'facecolor': 'white', 'alpha': 0.7},
                     transform=_ax.transAxes)
        else:
            # left
            _ax.legend(loc="upper left")
            _ax.text(0.03, 
                     0.10, 
                     my_anno_text, 
                     bbox={'facecolor': 'white', 'alpha': 0.7},
                     transform=_ax.transAxes) 
            
        if save_dir is not None:
            fig.savefig(save_dir / f"{team_name.lower().replace(' ', '_')}_territory_hist.png", dpi=150)
        
        return mu, sigma, dsigma, act, cdf
    except:
        print("")

def create_all_hists(
        day, 
        season=_SEASON, 
        save_dir=None
        ):
    leader_req = reqs.get(_BASE+"/stats/leaderboard",
                         params={"season": season,
                                 "day": day})
    leaders = leader_req.json()
    if day > 1:
        leader_req_yest = reqs.get(_BASE+"/stats/leaderboard",
                             params={"season": season,
                                     "day": day-1})
        leader_yest = leader_req_yest.json()
    
    mu = np.ones((len(leaders),))
    sig = np.ones((len(leaders),))
    dsig = np.ones((len(leaders),))
    act = np.ones((len(leaders),))
    for ind, leader in enumerate(leaders):
        print("Making hist for: ", leader["name"])
        if day > 1:
            prev_num_terry = [ll for ll in leader_yest if ll["name"] == leader["name"]][0]["territoryCount"]
        else: 
            prev_num_terry = leader["territoryCount"]
        try:
            mu[ind], sig[ind], dsig[ind], act[ind], cdf = create_expected_value_hist(
                leader["name"],
                day,
                int(prev_num_terry),
                season=season,
                save_dir=save_dir)
        except TypeError as inst:
            print("Unable to make hist for ", leader["name"], ". May not have any players today.")
            print(inst)
    
    return (min(dsig), leaders[np.argmin(dsig)]["name"]), (max(dsig), leaders[np.argmax(dsig)]["name"])

#%% Run Script with functions above
# HIT CTRL ENTER HERE TO RUN THE DAY'S DATA.
from pathlib import Path
import datetime
date = datetime.date
import os
SAVE_FLAG = True
REPLACE_FLAG = True

if SAVE_FLAG:
    figs_base_dir = Path(r"D:\Connor\Documents\GA 2022\Risk\cfb_artifacts")
    check_dir = figs_base_dir / f"{date.today().isoformat()}"
    # check_dir = figs_base_dir / "2020-04-22"
    asserted_dir = figs_base_dir / "temp_dir"
    # asserted_dir = check_dir
    if not check_dir.exists():
        os.mkdir(check_dir)
        save_dir = check_dir
    else:
        if REPLACE_FLAG:
            save_dir = check_dir
        else:
            save_dir = asserted_dir

dt_now = datetime.datetime.now()
deltaT = dt_now-datetime.datetime(2022, 1, 15)
day = deltaT.days
day = 3
# if dt_now.hour >= 14:
#     day += 1
# day = 1
# print(f"Generating plots for day={day}...")
mins_team, max_team = create_all_hists(day, save_dir=save_dir)
#%
# day=3
num_days = day
leader_req = reqs.get(_BASE+"/stats/leaderboard",
                          params={"season": _SEASON,
                                  "day": 1})
leaders = leader_req.json()

leader_list = [(leaders[i]["name"], [np.array([np.nan]*num_days), np.array([np.nan]*num_days), np.append(0, np.array([np.nan]*num_days))]) for i in range(len(leaders))]
team_dict = dict(leader_list)
for day in range(num_days, num_days+1):
    print(f"Generating plots for day={day}...")

    leader_req = reqs.get(_BASE+"/stats/leaderboard",
                          params={"season": _SEASON,
                                  "day": day})
    leaders = leader_req.json()
    
    leader_req_prev = reqs.get(_BASE+"/stats/leaderboard",
                          params={"season": _SEASON,
                                  "day": day-1})
    leaders_prev = leader_req_prev.json()
    
    leader_list = [(leaders[i]["name"], [np.array([np.nan]*num_days), np.array([np.nan]*num_days), np.append(0, np.array([np.nan]*num_days))]) for i in range(len(leaders))]
    team_dict = dict(leader_list)
    
    for ind, leader in enumerate(leaders):
        print("Making hist for: ", leader["name"])
        try:
            prev_data = [ll for ll in leaders_prev if ll["name"] == leader["name"]]
            mu, sig, dsig, act, cdf = create_expected_value_hist(
                leader["name"],
                day,
                int(prev_data[0]["territoryCount"]),
                season=_SEASON)
            
            prev_day = int(leader["territoryCount"])
            # scale the cdf output to some value between 0 and 1
            team_dict[leader["name"]][0][day-1] = cdf*2 - 1
            team_dict[leader["name"]][1][day-1] = dsig
            team_dict[leader["name"]][2][day] = act-prev_day
            
        except TypeError:
            print("Unable to make hist for ", leader["name"], ". May not have any players today.")
        plt.close()
        
#%
# step = 0.01
# x = np.arange(-1, 1+step, step)
# unif = np.ones(x.shape)
# out = np.copy(unif)
# for i in range(day):
#     out = np.convolve(out, unif)
    
# out = out / sum(out)
# x = np.linspace(-day, day, len(out))
# plt.plot(x, out)

#%%
# Run after dict is populated
# plt.close("all")
# team_req = reqs.get(_BASE+"/teams")
# team_info = team_req.json()

# # filter team_info to match what exists:
# for team in leader_list:
#     team_name = team[0]
#     team_uni, team_dsig, team_dt = np.copy(team_dict[team_name])
#     for info in team_info:
#         if info["name"] == team_name:
#             p_color = info["colors"]["primary"].strip()
#             s_color = info["colors"]["secondary"].strip()
#             p_color = tuple(float(val)/255 if ii < 3 else float(val) for ii, val in enumerate(p_color[5:-1].split(",")))
#             s_color = tuple(float(val)/255 if ii < 3 else float(val) for ii, val in enumerate(s_color[5:-1].split(",")))
            
#             if p_color[0:3] == (1, 1, 1):
#                 p_color = (0, 0, 0, p_color[3])
#                 color = p_color
#             else:
#                 if s_color[0:3] == (1, 1, 1):
#                     s_color = (0, 0, 0, s_color[3])
#                 color = s_color
#     style = "-"
#     # if team_name not in ['Alabama', 'Nebraska', 'Oklahoma', 'Stanford', 'Texas A&M', 'Wisconsin']:
#     #     style = "-"
#     # else:
#     #     if team_name in ["Wisconsin", "Stanford"]:
#     #         style = "--"
#     #     elif team_name in ["Texas A&M", "Oklahoma"]:
#     #         style = "-."
#     #     else:
#     #         style = "-"
#     fig101 = plt.figure(101, figsize=(12,7))
#     if sum(~np.isnan(team_dsig)) > 20:
#         plt.plot(np.arange(1, len(team_uni)+1), 
#                   np.cumsum(team_uni),
#                   color=color,
#                   linestyle=style,
#                   marker=".",
#                   markersize=6,
#                   label=team_name)
#     else:
#         plt.plot(np.arange(1, len(team_uni)+1), 
#                   np.cumsum(team_uni),
#                   color=color,
#                   linestyle=style,
#                   marker=".",
#                   markersize=6,
#                   alpha=0.7)
#     plt.legend(bbox_to_anchor=(1.02,0.5), loc="center left")
#     plt.title("Plot of $\sum_{n=1}^{day} (Actual_n - \mu_n)$")
#     plt.xlabel("Day")
#     plt.ylabel("Cumulative $(Actual_n - \mu_n)$")
#     plt.tight_layout()
    
#     fig102 = plt.figure(102, figsize=(12,7))
#     if sum(~np.isnan(team_dsig)) > 20:
#         plt.plot(np.arange(1, len(team_dsig)+1), 
#                   np.cumsum(team_dsig),
#                   color=color,
#                   linestyle=style,
#                   marker=".",
#                   markersize=6,
#                   label=team_name)
#     else:
#         plt.plot(np.arange(1, len(team_dsig)+1), 
#                   np.cumsum(team_dsig),
#                   color=color,
#                   linestyle=style,
#                   marker=".",
#                   markersize=6,
#                   alpha=0.7)
#     plt.legend(bbox_to_anchor=(1.02,0.5), loc="center left")
#     plt.title("Plot of $\sum_{n=1}^{day} \Delta\sigma_n$")
#     plt.xlabel("Day")
#     plt.ylabel("Cumulative $\Delta\sigma_n$")
#     plt.tight_layout()

#     fig103 = plt.figure(103, figsize=(12,7))
#     ax103 = plt.gca()
#     ax103.minorticks_on()
#     if sum(~np.isnan(team_dsig)) > 20:
#         plt.plot(np.arange(1, len(team_dt)+1), 
#                   np.cumsum(team_dt)+1,
#                   color=color,
#                   linestyle=style,
#                   marker=".",
#                   markersize=6,
#                   label=team_name)
#     else:
#         plt.plot(np.arange(1, len(team_dt)+1), 
#                   np.cumsum(team_dt)+1,
#                   color=color,
#                   linestyle=style,
#                   marker=".",
#                   markersize=6,
#                   alpha=0.7)
#     plt.legend(bbox_to_anchor=(1.02,0.5), loc="center left")
#     plt.title("Plot of $Territories_n$")
#     plt.xlabel("Day")
#     plt.ylabel("$Territories_n$")
#     plt.tight_layout()
#     plt.grid(True, which="major")
#     plt.grid(True, which="minor", color="#c6c6c6")
    
#     fig104 = plt.figure(104, figsize=(12,7))
#     ax104 = plt.gca()
#     plt.plot(x, 
#               out, 
#               color="#111111",
#               linestyle="-",
#               marker="",
#               alpha=1,
#               )
#     if sum(~np.isnan(team_dsig)) > 20:
#         yline(np.sum(team_uni),
#                   color=color,
#                   linestyle=style,
#                   marker=".",
#                   markersize=6,
#                   label=team_name)
#     else:
#         yline(np.sum(team_uni),
#                   color=color,
#                   linestyle=style,
#                   marker=".",
#                   markersize=6,
#                   alpha=0.7)
#     plt.legend(bbox_to_anchor=(1.02,0.5), loc="center left")
#     plt.title("Overall Luck Histogram")
#     plt.xlabel("Value")
#     plt.ylabel("Odds")
#     plt.tight_layout()
#     plt.grid(True, which="major")
#     plt.grid(True, which="minor", color="#c6c6c6")
   
# fig101.savefig(save_dir / "delta_exp_and_act_per_day.png", dpi=200)
# fig102.savefig(save_dir / "delta_sigma_per_day.png", dpi=200)
# fig103.savefig(save_dir / "territories_per_day.png", dpi=200)
# fig104.savefig(save_dir / "overall_luck_histogram.png", dpi=200)

# plt.figure(104, figsize=(12,7))
# if sum(~np.isnan(team_dsig)) > 20:
#     plt.plot(np.arange(1, len(team_dt)+1), 
#               team_dt,
#               color=color,
#               linestyle=style,
#               marker=".",
#               label=team_name)
# else:
#     plt.plot(np.arange(1, len(team_dt)+1), 
#               team_dt,
#               color=color,
#               linestyle=style,
#               marker=".",
#               alpha=0.7)
# plt.legend(bbox_to_anchor=(1.02,0.5), loc="center left")
# plt.title("Plot of $Territories_n$")
# plt.xlabel("Day")
# plt.ylabel("$Territories_n$")
# plt.tight_layout()
    
# Wanna make a MC sim to see the chance of a team getting +10 cumulative sigma
# or -10 cumulative sigma
# is this like, frequently going to happen? 
# Roll 50 turns of normal random variables 100,000 times for 100 teams
# see what the max and min of each "run" is and save that tuple
#%%
# import numpy as np
# num_runs = 100000

# max_vals = np.array([])
# min_vals = np.array([])

# for i in range(num_runs):
#     game = np.random.randn(10,50)
#     run = np.sum(game, axis=0)
#     run_max, run_min = np.max(run), np.min(run)
#     max_vals = np.append(max_vals, run_max)
#     min_vals = np.append(min_vals, run_min)
# #%%
# max_vals.sort()
# min_vals.sort()

# max_st = int(np.floor(max_vals[0]))
# max_end = int(np.ceil(max_vals[-1])+1)

# min_st = int(np.floor(min_vals[0]))
# min_end = int(np.ceil(min_vals[-1])+1)

# max_counts = np.array([])
# min_counts = np.array([])
# max_bins = np.array([])
# min_bins = np.array([])

# for i in range(max_st, max_end):
#     cnts = sum((max_vals < i+1) & (max_vals >= i))
#     max_counts = np.append(cnts, max_counts)
#     max_bins = np.append(i, max_bins)

# for i in range(min_st, min_end):
#     cnts = sum((min_vals < i+1) & (min_vals >= i))
#     min_counts = np.append(cnts, min_counts)
#     min_bins = np.append(i, min_bins)

# plt.figure()
# plt.bar(max_bins, max_counts / 1000)
# plt.figure()
# plt.bar(min_bins, min_counts / 1000)