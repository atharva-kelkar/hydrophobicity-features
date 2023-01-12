"""
plotting.py
This script contains tools to plot ML data

CREATED ON: 10/16/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
### IMPORT MODULES
##############################################################################
## IMPORT NUMPY
import numpy as np
## IMPORT MATPLOTLIB
import matplotlib.pyplot as plt
## IMPORT PLOTTING FUNCTION
from sam_analysis.plotting.jacs_single import JACS

##############################################################################
## FUNCTIONS
##############################################################################
## FUNCTION TO CREATE PARITY PLOT
def plot_parity( x, y, 
                 xerr         = [],
                 yerr         = [],
                 title        = None,
                 xlabel       = None,
                 ylabel       = None,
                 xticks       = [],
                 yticks       = [],
                 markers      = None,
                 colors       = None,
                 point_labels = None,
                 fig_path     = None, ):
    ## PRINT
    print( "\n--- CREATING PARITY PLOT ---")
    
    ## SET PLOT DEFAULT
    plot_details = JACS()
    
    ## CREATE SUBPLOTS
    fig, ax = plt.subplots()
    fig.subplots_adjust( left = 0.15, bottom = 0.15, right = 0.95, top = 0.90 )
    
    ## SET TITLE
    if title is not None:
        ax.set_title( title )

    ## MAKE NONE LIST
    if point_labels is None:
        point_labels = len(y) * [ None ]

    ## GET MARKERS
    if markers is None:
        markers = len(y) * [ "s" ]

    ## SET X AND Y LABELS
    if xlabel is not None:
        ax.set_xlabel( xlabel )
    if ylabel is not None:
        ax.set_ylabel( ylabel )
    
    ## PUT Y IN LIST IF NOT
    if type(y) is not list:
        ## GENERATE ZEROS FOR ERR IF NONE
        if len(xerr) < 1:
            xerr = np.zeros_like(x)
        if len(yerr) < 1:
            yerr = np.zeros_like(y)
        ## PUT Y DATA IN LISTS
        y = [ y ]
        yerr = [ yerr ]
    else:
        ## GENERATE ZEROS FOR ERR IF NONE
        if len(xerr) < 1:
            xerr = []
            for xx in x:
                xerr.append( np.zeros_like(xx) )
        if len(yerr) < 1:
            yerr = []
            for yy in y:
                yerr.append( np.zeros_like(yy) )
    
    ## GET COLOR MAP
    if colors is None:
        colors = [ plot_details.colormap(ii) for ii in np.linspace( 0.0, 1.0, len(y) ) ]
    elif type(colors) is int: 
        colors = [ plot_details.colormap(ii) for ii in np.linspace( 0.0, 1.0, colors ) ] * ( len(y)//colors )
    else:
        colors = colors

    ## PLOT X-Y LINE
    ax.plot( [ 0, 130 ], [ 0, 130 ],
             linestyle = ":",
             linewidth = 1.5,
             color = "darkgray" )

    ## LOOP THROUGH Y TYPES
    for ii, (xx, xxerr, yy, yyerr) in enumerate(zip( x, xerr, y, yerr )):        
        ## PLOT POINTS WITH ERROR BARS
        ax.errorbar( xx, yy,
                     xerr       = xxerr,
                     yerr       = yyerr,
                     linestyle  = "None",
                     marker     = markers[ii],
                     markersize = 6.,
                     color      = colors[ii],
                     ecolor     = colors[ii],
                     elinewidth = 0.5, 
                     capsize    = 2., )
        ## PLOT WITH SCATTER FOR BETTER LEGEND
        ax.scatter( [], [],
                    marker = markers[ii],
                    s      = 6**2,
                    color  = colors[ii],
                    label  = point_labels[ii] )

    ## SET X AND Y TICKS
    if len(xticks) > 1:
        x_min   = xticks[0]
        x_max   = xticks[1]
        x_diff  = xticks[2]
        x_lower = x_min - 0.5*x_diff
        x_upper = x_max + 0.5*x_diff
        ax.set_xticks( np.arange( x_min, x_max + x_diff, x_diff ), minor = False )       # sets major ticks
        ax.set_xticks( np.arange( x_lower, x_max + 1.5*x_diff, x_diff ), minor = True )
        ax.set_xlim( x_lower, x_upper )
        
    if len(yticks) > 1:
        y_min   = yticks[0]
        y_max   = yticks[1]
        y_diff  = yticks[2]
        y_lower = y_min - 0.5*y_diff
        y_upper = y_max + 0.5*y_diff
        ax.set_yticks( np.arange( y_min, y_max + y_diff, y_diff ), minor = False )       # sets major ticks
        ax.set_yticks( np.arange( y_lower, y_max + 1.5*y_diff, y_diff ), minor = True )
        ax.set_ylim( y_lower, y_upper )

    ## ADD LEGEND
    if len(point_labels) > 1:
        ax.legend( loc = 'best', ncol = 3, fontsize = 6 )
        
    fig.set_size_inches( plot_details.width, plot_details.width ) # 1:1 aspect ratio
    if fig_path is not None:
        print( "FIGURE SAVED TO: %s" % fig_path )
        fig.savefig( fig_path + ".png", dpi = 300, facecolor = 'w', edgecolor = 'w' )
        fig.savefig( fig_path + ".svg", dpi = 300, facecolor = 'w', edgecolor = 'w' )


## CREATE VALIDATION PLOT
def plot_validation( x, y, 
                     yerr        = [],
                     title       = None,
                     xlabel      = None,
                     ylabel      = None,
                     xticks      = [],
                     yticks      = [],
                     line_labels = None,
                     fig_path    = None, ):
    ## PRINT
    print( "\n--- CREATING VALIDATION PLOT ---")
    ## SET PLOT DEFAULT
    plot_details = JACS()
    
    ## CREATE SUBPLOTS
    fig, ax = plt.subplots()
    fig.subplots_adjust( left = 0.15, bottom = 0.15, right = 0.95, top = 0.90 )
    
    ## SET TITLE
    if title is not None:
        ax.set_title( title )
    
    ## SET X AND Y LABELS
    if xlabel is not None:
        ax.set_xlabel( xlabel )
    if ylabel is not None:
        ax.set_ylabel( ylabel )

    ## MAKE NONE LIST
    if line_labels is None:
        line_labels = len(y) * [ None ]
    
    ## PUT Y IN LIST IF NOT
    if type(y) is not list:
        ## GENERATE ZEROS FOR ERR IF NONE
        if len(yerr) < 1:
            yerr = np.zeros_like(y)
        ## PUT Y DATA IN LISTS
        y = [ y ]
        yerr = [ yerr ]
    else:
        ## GENERATE ZEROS FOR ERR IF NONE
        if len(yerr) < 1:
            yerr = []
            for yy in y:
                yerr.append( np.zeros_like(yy) )

    ## GET COLOR MAP
    colors = [ plot_details.colormap(ii) for ii in np.linspace( 0.0, 1.0, len(y) ) ]
    
    ## LOOP THROUGH Y TYPES
    for ii, (yy, yyerr) in enumerate(zip( y, yerr )):            
        ## PLOT LINES
        plt.plot( x,
                  yy,
                  linestyle = '-',
                  linewidth = 1.5,
                  color     = colors[ii],
                  label     = line_labels[ii] )
        
        ## PLOT SHADED ERROR
        plt.plot( x,
                  yy + yyerr,
                  linestyle = '-',
                  linewidth = 1.0,
                  color     = colors[ii] )
        plt.plot( x,
                  yy - yyerr,
                  linestyle = '-',
                  linewidth = 1.0,
                  color     = colors[ii] )
        plt.fill_between( x,
                          yy + yyerr,
                          yy - yyerr,
                          color = colors[ii],
                          alpha = 0.5, )    
    
    ## SET X AND Y TICKS
    if len(xticks) > 1:
        x_min   = xticks[0]
        x_max   = xticks[1]
        x_diff  = xticks[2]
        x_lower = x_min - 0.5*x_diff
        x_upper = x_max + 0.5*x_diff
        ax.set_xticks( np.arange( x_min, x_max + x_diff, x_diff ), minor = False )       # sets major ticks
        ax.set_xticks( np.arange( x_lower, x_max + 1.5*x_diff, x_diff ), minor = True )
        ax.set_xlim( x_lower, x_upper )
        
    if len(yticks) > 1:
        y_min   = yticks[0]
        y_max   = yticks[1]
        y_diff  = yticks[2]
        y_lower = y_min - 0.5*y_diff
        y_upper = y_max + 0.5*y_diff
        ax.set_yticks( np.arange( y_min, y_max + y_diff, y_diff ), minor = False )       # sets major ticks
        ax.set_yticks( np.arange( y_lower, y_max + 1.5*y_diff, y_diff ), minor = True )
        ax.set_ylim( y_lower, y_upper )

    if None not in line_labels:
        ax.legend( loc = 'best', ncol = 1 )
        
    fig.set_size_inches( plot_details.width, plot_details.height ) # 4:3 aspect ratio
    if fig_path is not None:
        print( "FIGURE SAVED TO: %s" % fig_path )
        fig.savefig( fig_path + ".png", dpi = 300, facecolor = 'w', edgecolor = 'w' )
        fig.savefig( fig_path + ".svg", dpi = 300, facecolor = 'w', edgecolor = 'w' )
    
## FUNCTION TO CREATE SALIENCY PLOT
def plot_saliency( y,
                   yerr      = [],
                   title     = None,
                   ylabel    = None,
                   xticks    = [],
                   yticks    = [],
                   labels    = [],
                   fig_path  = None, ):
    ## PRINT
    print( "\n--- CREATING SALIENCY PLOT ---")
    
    ## SET PLOT DEFAULT
    plot_details = JACS()
    
    ## GET X ARRAY
    x = np.arange( 0, len(y), 1 )
    
    ## CREATE SUBPLOTS
    fig, ax = plt.subplots()
    fig.subplots_adjust( left = 0.20, bottom = 0.40, right = 0.95, top = 0.95 )
    
    ## SET TITLE
    if title is not None:
        ax.set_title( title )
    
    ## SET Y LABELS
    if ylabel is not None:
        ax.set_ylabel( ylabel )
    
    ## GENERATE ZEROS FOR ERR IF NONE
    if len(yerr) < 1:
        yerr = np.zeros_like(y)
    
    ## PLOT ZERO LINE
    ax.plot( [ x[0]-1, x[-1]+1 ], [ 0, 0 ], 
             color = "black", 
             linestyle = "-", 
             linewidth = 0.5 )
        
    ## PLOT BAR
    ax.bar( x, y,
            linestyle = "None",
            color     = "grey",
            edgecolor = "black",
            yerr      = yerr,
            ecolor    = "black",
            linewidth = 0.5,
            capsize   = 2.0, )    
           
    ## SET X TICKS
    x_tick = 1
    ## GET X MAJOR TICKS
    x_ticks = np.arange( 0., len(labels), 1 )
    ## GET X MINOR TICKS
    x_minor_ticks = np.concatenate(( x_ticks - 0.5*x_tick, np.array([ x_ticks[-1] + 0.5*x_tick ]) ))
    ## SET X TICKS
    ax.set_xlim( x_minor_ticks[0], x_minor_ticks[-1] )
    ax.set_xticks( x_ticks, minor = False )       # sets major ticks
    ax.set_xticks( x_minor_ticks, minor = True )  # sets minor ticks
    if len(labels) > 1:
        ax.set_xticklabels( labels, rotation = 90 ) # sets tick labels
    else:
        ax.set_xlabel( "Feature ID" )
    
    ## Y TICKS        
    if len(yticks) > 1:
        y_min   = yticks[0]
        y_max   = yticks[1]
        y_diff  = yticks[2]
        y_lower = y_min - 0.5*y_diff
        y_upper = y_max + 0.5*y_diff
        ax.set_yticks( np.arange( y_min, y_max + y_diff, y_diff ), minor = False )       # sets major ticks
        ax.set_yticks( np.arange( y_lower, y_max + 1.5*y_diff, y_diff ), minor = True )
        ax.set_ylim( y_lower, y_upper )
        
    fig.set_size_inches( plot_details.width, plot_details.height+1.5 ) # 4:3 aspect ratio
    if fig_path is not None:
        print( "FIGURE SAVED TO: %s" % fig_path )
        fig.savefig( fig_path + ".png", dpi = 300, facecolor = 'w', edgecolor = 'w' )
        fig.savefig( fig_path + ".svg", dpi = 300, facecolor = 'w', edgecolor = 'w' )

## FUNCTION TO CREATE SALIENCY PLOT
def plot_multi_saliency( y,
                         yerr      = [],
                         title     = None,
                         ylabel    = None,
                         yticks    = [],
                         labels    = [],
                         fig_path  = None, ):
    ## PRINT
    print( "\n--- CREATING SALIENCY PLOT ---")
    
    ## SET PLOT DEFAULT
    plot_details = JACS()
    
    ## GET X ARRAY
    x = np.arange( 0, len(y[0,:]), 1 )
    
    ## CREATE SUBPLOTS
    fig, axs = plt.subplots( nrows = len(y), sharex = True )
    fig.subplots_adjust( left = 0.20, bottom = 0.30, right = 0.95, top = 0.95 )
    
    for ii, ax in enumerate( axs ):
        ## SET TITLE
        if title is not None:
            ax.set_title( title )
        
        ## SET Y LABELS
        if ylabel is not None:
            ax.set_ylabel( ylabel )
        
        ## GENERATE ZEROS FOR ERR IF NONE
        if len(yerr) < 1:
            yerr = np.zeros_like(y[ii,:])
        
        ## PLOT ZERO LINE
        ax.plot( [ x[0]-1, x[-1]+1 ], [ 0, 0 ], 
                 color = "black", 
                 linestyle = "-", 
                 linewidth = 0.5 )
            
        ## PLOT BAR
        ax.bar( x,
                y[ii,:],
                linestyle = "None",
                color     = "grey",
                edgecolor = "black",
                yerr      = yerr[ii,:],
                ecolor    = "black",
                linewidth = 0.5,
                capsize   = 2.0, )
        
        ## Y TICKS        
        if len(yticks) > 1:
            y_min   = yticks[0]
            y_max   = yticks[1]
            y_diff  = yticks[2]
            y_lower = y_min - 0.5*y_diff
            y_upper = y_max + 0.5*y_diff
            ax.set_yticks( np.arange( y_min, y_max + y_diff, y_diff ), minor = False )       # sets major ticks
            ax.set_yticks( np.arange( y_lower, y_max + 1.5*y_diff, y_diff ), minor = True )
            ax.set_ylim( y_lower, y_upper )

    ## SET X TICKS
    x_tick = 1
    ## GET X MAJOR TICKS
    x_ticks = np.arange( 0., len(labels), 1 )
    ## GET X MINOR TICKS
    x_minor_ticks = np.concatenate(( x_ticks - 0.5*x_tick, np.array([ x_ticks[-1] + 0.5*x_tick ]) ))
    ## SET X TICKS
    ax.set_xlim( x_minor_ticks[0], x_minor_ticks[-1] )
    ax.set_xticks( x_ticks, minor = False )       # sets major ticks
    ax.set_xticks( x_minor_ticks, minor = True )  # sets minor ticks
    if len(labels) > 1:
        ax.set_xticklabels( labels, rotation = 90 ) # sets tick labels
    else:
        ax.set_xlabel( "Feature ID" )
        
    fig.set_size_inches( plot_details.width, plot_details.height+3.5 ) # 4:3 aspect ratio
    if fig_path is not None:
        print( "FIGURE SAVED TO: %s" % fig_path )
        fig.savefig( fig_path + ".png", dpi = 300, facecolor = 'w', edgecolor = 'w' )
        fig.savefig( fig_path + ".svg", dpi = 300, facecolor = 'w', edgecolor = 'w' )
