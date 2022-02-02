# PyRisk

A repo of Python Risk tools

## General Use: Making the Histogram plots

First, make sure you have the following modules installed.
The recommended versions are mentioned as well, but any version greater than 
that should suffice.

* scipy.__version__: '1.6.2'

* numpy.__version__: '1.20.1'

* matplotlib.__version__: '3.3.4'

* reqs.__version__: '2.25.1'

### Clone the repo, and find cfb_plots.py

Within the repo, there is a record of all plots checked in for each day, as 
well as the `cfb_plots.py` file, which can be used to generate any given day's
plots again.  The script takes a day input (within the if statement below) to
scrape the API for to produce the plots.

``` python
if __name__ == "__main__":
    day = 15
    main(day)
```

### Make an output directory

Now that you found the script, its time to edit that `output_directory` term.
Make it whatever you wish, but make sure to change that otherwise the script 
will error.


### Prepping the script for the new day data

Within that section, there should be a "day" input set as an int. Increment 
that, then run the script to make the plots go brr.

Ignore all warnings, they should be okay, they are just due to not protecting 
cases where there is only 1 bin, such as early season, or when a team is on 
their last leg. (I'll fix them later, that's what I get for having sloppy code)

### Posting the album

And that's it. To make the posts, I post them onto Imgur as an album. I'll add 
a screenshot of the current board at the end, and the day priors at the start.

I currently manually take screen grabs afterwards to add to the post, then I 
just have fun with the write up myself, but post them however you'd like, even 
a dull post of just the charts is better than no post (I like it, cause I like 
to think it adds game interactions and interest overall).
