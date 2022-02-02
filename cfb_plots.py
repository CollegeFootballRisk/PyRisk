# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 15:46:47 2022

@author: Connor
"""

#
# Imports
#
import requests as reqs
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

from pathlib import Path
from scipy.special import erf

_BASE ="https://collegefootballrisk.com/api"
_SEASON = 1
plt.style.use("bmh")

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
        rank,
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
            fig.savefig(save_dir / f"{rank}_{team_name.lower().replace(' ', '_')}_territory_hist.png", dpi=150)
        
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
                leader["rank"],
                day,
                int(prev_num_terry),
                season=season,
                save_dir=save_dir)
        except TypeError as inst:
            print("Unable to make hist for ", leader["name"], ". May not have any players today.")
            print(inst)
    
    return (min(dsig), leaders[np.argmin(dsig)]["name"]), (max(dsig), leaders[np.argmax(dsig)]["name"])

def main(day=None):
    date = datetime.date
    # Set this true if you want to save the graphs
    SAVE_FLAG = True
    # Set this true if you want to replace the current existing graphs
    REPLACE_FLAG = True
    
    if SAVE_FLAG:
        output_directory = r"D:\Connor\Documents\GA 2022\Risk\cfb_artifacts"
        figs_base_dir = Path(output_directory)
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
    
    # Get delta Time since start of game
    if not day:
        dt_now = datetime.datetime.now()
        deltaT = dt_now-datetime.datetime(2022, 1, 15)
        day = deltaT.days  # get just the delta number of days
    
    print(f"Generating plots for day = {day}...")
    mins_team, max_team = create_all_hists(day, save_dir=save_dir)

if __name__ == "__main__":
    day = 15
    main(day)