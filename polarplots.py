#  polarplots.py an image generator extension for weewx
#  (c) 2021 Michel Anders
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  version 202103041037

"""Generate polar plots for various weewx data with a directional component"""

import datetime
import logging
import os.path
import time
import sys
from pathlib import Path

sys.path.append('/usr/share/weewx')

import weeplot.genplot
from weeplot.genplot import int2rgbstr
import weeplot.utilities
import weeutil.logger
import weeutil.weeutil
import weewx.reportengine
import weewx.units
import weewx.xtypes
import weewx.station
from weeutil.config import search_up, accumulateLeaves
from weeutil.weeutil import to_bool, to_int, to_float, TimeSpan
from weewx.units import ValueTuple

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mticker

import numpy as np

log = logging.getLogger(__name__)

blue2red = ListedColormap(np.dstack((np.linspace(0,1,256), np.zeros(256), np.linspace(1,0,256)))[0] )

class PolarPlotLine:
    
    """Represents a single line (or bar) in a plot. """
    def __init__(self, x, y, label='', color=None, fill_color=None, width=None, plot_type='line',
                 line_type='solid', marker_type=None, marker_size=10, 
                 bar_width=None, vector_rotate = None, gap_fraction=None, theta_bins=0):
        self.x               = x
        self.y               = y
        self.label           = label
        self.plot_type       = plot_type
        self.line_type       = line_type
        self.marker_type     = marker_type
        self.marker_size     = marker_size
        self.color           = color
        self.fill_color      = fill_color
        self.width           = width
        self.bar_width       = bar_width
        self.vector_rotate   = vector_rotate
        self.gap_fraction    = gap_fraction
        self.theta_bins      = theta_bins
    
    def __str__(self):
        return str(self.x) + str(self.y)
        
class PolarPlot(weeplot.genplot.GeneralPlot):

    def _renderDayNight(self, tmin, tmax, ax):
        # get day/night transitions
        (first, transitions) = weeutil.weeutil.getDayNightTransitions(
                tmin, tmax, self.latitude, self.longitude)
        # construct the colormap on the range [0,1]
        colors = []
        period_length_s = tmax - tmin
        color_step_s = period_length_s/256
        color = self.daynight_day_color if first == 'day' else self.daynight_night_color
        xstart = tmin
        for x in transitions:
            color_steps = int(((x - xstart)/color_step_s)+0.5)
            colors.extend([int2rgbstr(color)] * color_steps)
            color = self.daynight_night_color if color == self.daynight_day_color else self.daynight_day_color
            xstart = x
        color_steps = int(((tmax - xstart)/color_step_s)+0.5)
        colors.extend([f"#{color:06x}"] * color_steps)
        color_map = ListedColormap(colors)
        # draw colored background
        rad = np.linspace(1, 0, 100)
        azm = np.linspace(0, 2 * np.pi, 100)
        dayr, dayθ = np.meshgrid(rad, azm)
        # reversed colormap because generated map is from older to more recent but we will plot with more recent closer to the center 
        ax.pcolormesh(dayθ, dayr, dayr, cmap=color_map.reversed(), shading='gouraud', zorder=0)

    def render(self):
        print('PolarPlot render')
        
        # x are timestamps
        # y are angles (in degrees)
        tmin = np.min(self.line_list[0].x)  # this might be in the Xscale info already ...
        tmax = np.max(self.line_list[0].x)
        print(self.image_background_color, self.chart_background_color)
        #print(self.bottom_label_font_path, self.bottom_label_font_size)
        #print(type(self.line_list[0].x[0]),type(self.line_list[0].y[0]))
        self.fig = plt.figure(facecolor=int2rgbstr(self.chart_background_color))
        self.ax = self.fig.add_subplot(111, projection='polar', facecolor=int2rgbstr(self.image_background_color))
        self.ax.set_xlabel(self.bottom_label, fontproperties=Path(self.bottom_label_font_path) if self.bottom_label_font_path else None)
        self.ax.set_rticks([0.0, 0.25, 0.5, 0.75])
        # just setting fixed labels results in a user warning to we explicitly set a fixed locator first
        ticks_loc = self.ax.get_xticks().tolist()
        self.ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        self.ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        # make an annulus (gap near the center)
        self.ax.set_rorigin(-0.25)
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        print('theta_bins', self.line_list[0].theta_bins)
        if self.line_list[0].plot_type == 'vector':
            t = np.array(self.line_list[0].x, dtype=np.float32)
            v = np.conj(np.array(self.line_list[0].y, dtype=np.complex64))
            θ = np.angle(v) + np.pi/2
            a = np.abs(self.line_list[0].y)
            r = (t - tmin)/(tmax-tmin)
            c = self.ax.scatter(θ, 1 - r, c=a, cmap=blue2red, vmax=12)  # more recent is closer to the center
        else:
            if self.show_daynight:
                self._renderDayNight(tmin, tmax, self.ax)
            t = np.array(self.line_list[0].x, dtype=np.float32)
            direction = np.array(self.line_list[0].y, dtype=np.float32)
            r = (t - tmin)/(tmax-tmin)
            θ = np.radians(direction)
            theta_bins = int(self.line_list[0].theta_bins)
            if  theta_bins > 0:
                half_bin_width = np.pi / theta_bins 
                tbins = np.linspace(0,2*np.pi,theta_bins+1)
                rbins = np.linspace(0,1,8)  # nbins hardcoded for now
                H, θedges, redges = np.histogram2d(θ + half_bin_width % 2*np.pi, r, bins=[tbins, rbins])
                H = np.append(H,H[0].reshape(1,-1), axis=0)
                tbins -= half_bin_width
                offset = 0
                print(H)
                for rbin in range(len(rbins)-1):
                    counts = H[:,rbin] + offset
                    offset += counts
                    print(tbins.shape, counts.shape)
                    patch = self.ax.plot(tbins,counts)
            else:
                c = self.ax.scatter(θ, 1-r)   # more recent is closer to the center
        self.ax.grid(True)  # needs to come after call pcolormesh (because that fie sets it to False)
        return self.fig


# =============================================================================
#                    Class PolarPlotGenerator
#
# Large parts were copied from the original ImageGenerator class but because
# that class was rather monolithic it made hardly any sense to inherit from it.
# =============================================================================

class PolarPlotGenerator(weewx.reportengine.ReportGenerator):
    """Class for managing the image generator."""

    def run(self):
        self.setup()
        self.genImages(self.gen_ts)

    def setup(self):
        try:
            d = self.skin_dict['Labels']['Generic']
        except KeyError:
            d = {}
        self.title_dict = weeutil.weeutil.KeyDict(d)
        self.image_dict = self.skin_dict['PolarPlotGenerator']
        self.formatter  = weewx.units.Formatter.fromSkinDict(self.skin_dict)
        self.converter  = weewx.units.Converter.fromSkinDict(self.skin_dict)
        # ensure that the skin_dir is in the image_dict
        self.image_dict['skin_dir'] = os.path.join(
            self.config_dict['WEEWX_ROOT'],
            self.skin_dict['SKIN_ROOT'],
            self.skin_dict['skin'])
        # ensure that we are in a consistent right location
        os.chdir(self.image_dict['skin_dir'])

    def genImages(self, gen_ts):
        """Generate the images.

        The time scales will be chosen to include the given timestamp, with
        nice beginning and ending times.

        gen_ts: The time around which plots are to be generated. This will
        also be used as the bottom label in the plots. [optional. Default is
        to use the time of the last record in the database.]
        """
        t1 = time.time()
        ngen = 0

        # determine how much logging is desired
        log_success = to_bool(search_up(self.image_dict, 'log_success', True))

        # Loop over each time span class (day, week, month, etc.):
        for timespan in self.image_dict.sections:

            # Now, loop over all plot names in this time span class:
            for plotname in self.image_dict[timespan].sections:

                # Accumulate all options from parent nodes:
                plot_options = accumulateLeaves(self.image_dict[timespan][plotname])

                plotgen_ts = gen_ts
                if not plotgen_ts:
                    binding = plot_options['data_binding']
                    db_manager = self.db_binder.get_manager(binding)
                    plotgen_ts = db_manager.lastGoodStamp()
                    if not plotgen_ts:
                        plotgen_ts = time.time()

                image_root = os.path.join(self.config_dict['WEEWX_ROOT'],
                                          plot_options['HTML_ROOT'])
                # Get the path that the image is going to be saved to:
                img_file = os.path.join(image_root, '%s.png' % plotname)

                ai = to_int(plot_options.get('aggregate_interval'))
                # Check whether this plot needs to be done at all:
                if skipThisPlot(plotgen_ts, ai, img_file):
                    continue

                # skip image files that are fresh, but only if staleness is defined
                stale = to_int(plot_options.get('stale_age'))
                if stale is not None:
                    t_now = time.time()
                    try:
                        last_mod = os.path.getmtime(img_file)
                        if t_now - last_mod < stale:
                            log.debug("Skip '%s': last_mod=%s age=%s stale=%s",
                                      img_file, last_mod, t_now - last_mod, stale)
                            continue
                    except os.error:
                        pass

                # Create the subdirectory that the image is to be put in.
                # Wrap in a try block in case it already exists.
                try:
                    os.makedirs(os.path.dirname(img_file))
                except OSError:
                    pass

                # Create a new instance of a polar plot and start adding to it
                plot = PolarPlot(plot_options)

                # Calculate a suitable min, max time for the requested time.
                (minstamp, maxstamp, timeinc) = weeplot.utilities.scaletime(plotgen_ts - int(plot_options.get('time_length', 86400)), plotgen_ts)
                # Override the x interval if the user has given an explicit interval:
                timeinc_user = to_int(plot_options.get('x_interval'))
                if timeinc_user is not None:
                    timeinc = timeinc_user
                plot.setXScaling((minstamp, maxstamp, timeinc))

                # Set the y-scaling, using any user-supplied hints:
                plot.setYScaling(weeutil.weeutil.convertToFloat(plot_options.get('yscale', ['None', 'None', 'None'])))

                # Get a suitable bottom label:
                bottom_label_format = plot_options.get('bottom_label_format', '%m/%d/%y %H:%M')
                bottom_label = time.strftime(bottom_label_format, time.localtime(plotgen_ts))
                plot.setBottomLabel(bottom_label)

                # Set day/night display
                if self.stn_info is not None:
                    plot.setLocation(self.stn_info.latitude_f, self.stn_info.longitude_f)
                    plot.setDayNight(to_bool(plot_options.get('show_daynight', False)),
                                    weeplot.utilities.tobgr(plot_options.get('daynight_day_color', '0xffffff')),
                                    weeplot.utilities.tobgr(plot_options.get('daynight_night_color', '0xf0f0f0')),
                                    weeplot.utilities.tobgr(plot_options.get('daynight_edge_color', '0xefefef')))

                # Loop over each line to be added to the plot.
                for line_name in self.image_dict[timespan][plotname].sections:

                    # Accumulate options from parent nodes.
                    line_options = accumulateLeaves(self.image_dict[timespan][plotname][line_name])

                    # See what SQL variable type to use for this line. By
                    # default, use the section name.
                    var_type = line_options.get('data_type', line_name)

                    # Look for aggregation type:
                    aggregate_type = line_options.get('aggregate_type')
                    if aggregate_type in (None, '', 'None', 'none'):
                        # No aggregation specified.
                        aggregate_type = aggregate_interval = None
                    else:
                        try:
                            # Aggregation specified. Get the interval.
                            aggregate_interval = line_options.as_int('aggregate_interval')
                        except KeyError:
                            log.error("Aggregate interval required for aggregate type %s", aggregate_type)
                            log.error("Line type %s skipped", var_type)
                            continue

                    # Now its time to find and hit the database:
                    binding = line_options['data_binding']
                    db_manager = self.db_binder.get_manager(binding)
                    start_vec_t, stop_vec_t ,data_vec_t = weewx.xtypes.get_series(var_type,
                                                                                  TimeSpan(minstamp, maxstamp),
                                                                                  db_manager,
                                                                                  aggregate_type=aggregate_type,
                                                                                  aggregate_interval=aggregate_interval)

                    # Get the type of plot ("bar', 'line', or 'vector')
                    plot_type = line_options.get('plot_type', 'line')

                    if aggregate_type and aggregate_type.lower() in ('avg', 'max', 'min') and plot_type != 'bar':
                        # Put the point in the middle of the aggregate_interval for these aggregation types
                        start_vec_t = ValueTuple([x - aggregate_interval / 2.0 for x in start_vec_t[0]],
                                                 start_vec_t[1], start_vec_t[2])
                        stop_vec_t = ValueTuple([x - aggregate_interval / 2.0 for x in stop_vec_t[0]],
                                                stop_vec_t[1], stop_vec_t[2])

                    # Do any necessary unit conversions:
                    new_start_vec_t = self.converter.convert(start_vec_t)
                    new_stop_vec_t  = self.converter.convert(stop_vec_t)
                    new_data_vec_t = self.converter.convert(data_vec_t)

                    # Add a unit label. NB: all will get overwritten except the
                    # last. Get the label from the configuration dictionary.
                    unit_label = line_options.get('y_label', weewx.units.get_label_string(self.formatter, self.converter, var_type))
                    # Strip off any leading and trailing whitespace so it's
                    # easy to center
                    plot.setUnitLabel(unit_label.strip())

                    # See if a line label has been explicitly requested:
                    label = line_options.get('label')
                    if not label:
                        # No explicit label. Look up a generic one. NB: title_dict is a KeyDict which
                        # will substitute the key if the value is not in the dictionary.
                        label = self.title_dict[var_type]

                    # See if a color has been explicitly requested.
                    color = line_options.get('color')
                    if color is not None: color = weeplot.utilities.tobgr(color)
                    fill_color = line_options.get('fill_color')
                    if fill_color is not None: fill_color = weeplot.utilities.tobgr(fill_color)

                    # Get the line width, if explicitly requested.
                    width = to_int(line_options.get('width'))

                    interval_vec = None
                    gap_fraction = None

                    # Some plot types require special treatments:
                    if plot_type == 'vector':
                        vector_rotate_str = line_options.get('vector_rotate')
                        vector_rotate = -float(vector_rotate_str) if vector_rotate_str is not None else None
                    else:
                        vector_rotate = None

                        if plot_type == 'bar':
                            interval_vec = [x[1] - x[0]for x in zip(new_start_vec_t.value, new_stop_vec_t.value)]
                        elif plot_type == 'line':
                            gap_fraction = to_float(line_options.get('line_gap_fraction'))
                        if gap_fraction is not None:
                            if not 0 < gap_fraction < 1:
                                log.error("Gap fraction %5.3f outside range 0 to 1. Ignored.", gap_fraction)
                                gap_fraction = None

                    # Get the type of line (only 'solid' or 'none' for now)
                    line_type = line_options.get('line_type', 'solid')
                    if line_type.strip().lower() in ['', 'none']:
                        line_type = None

                    marker_type = line_options.get('marker_type')
                    marker_size = to_int(line_options.get('marker_size', 8))
                    
                    theta_bins = plot_options.get('theta_bins',0)
                    # Add the line to the emerging plot:
                    plot.addLine(PolarPlotLine(
                        new_stop_vec_t[0], new_data_vec_t[0],
                        label         = label,
                        color         = color,
                        fill_color    = fill_color,
                        width         = width,
                        plot_type     = plot_type,
                        line_type     = line_type,
                        marker_type   = marker_type,
                        marker_size   = marker_size,
                        bar_width     = interval_vec,
                        vector_rotate = vector_rotate,
                        gap_fraction  = gap_fraction,
                        theta_bins    = theta_bins))

                # OK, the plot is ready. Render it onto an image
                image = plot.render()

                try:
                    # Now save the image
                    image.savefig(img_file, bbox_inches='tight')
                    ngen += 1
                except IOError as e:
                    log.error("Unable to save to file '%s' %s:", img_file, e)
        t2 = time.time()

        if log_success:
            log.info("Generated %d images for report %s in %.2f seconds", ngen, self.skin_dict['REPORT_NAME'], t2 - t1)

def skipThisPlot(time_ts, aggregate_interval, img_file):
    """A plot can be skipped if it was generated recently and has not changed.
    This happens if the time since the plot was generated is less than the
    aggregation interval."""

    # Images without an aggregation interval have to be plotted every time.
    # Also, the image definitely has to be generated if it doesn't exist.
    if aggregate_interval is None or not os.path.exists(img_file):
        return False

    # If its a very old image, then it has to be regenerated
    if time_ts - os.stat(img_file).st_mtime >= aggregate_interval:
        return False

    # Finally, if we're on an aggregation boundary, regenerate.
    time_dt = datetime.datetime.fromtimestamp(time_ts)
    tdiff = time_dt -  time_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return abs(tdiff.seconds % aggregate_interval) > 1

def build_skin_dict(config_dict, report):
    """Find and build the skin_dict for the given report"""

    # Start with the defaults in the defaults module. Because we will be modifying it, we need
    # to make a deep copy.
    skin_dict = weeutil.config.deep_copy(weewx.defaults.defaults)

    # Add the report name:
    skin_dict['REPORT_NAME'] = report

    # Now add the options in the report's skin.conf file. Start by figuring where it is located.
    skin_config_path = os.path.join(
        config_dict['WEEWX_ROOT'],
        config_dict['StdReport']['SKIN_ROOT'],
        config_dict['StdReport'][report].get('skin', ''),
        'skin.conf')

    # Now retrieve the configuration dictionary for the skin. Wrap it in a try block in case we fail.  It is ok if
    # there is no file - everything for a skin might be defined in the weewx configuration.
    try:
        merge_dict = configobj.ConfigObj(skin_config_path, file_error=True, encoding='utf-8')
        log.debug("Found configuration file %s for report '%s'", skin_config_path, report)
        # Merge the skin config file in:
        weeutil.config.merge_config(skin_dict, merge_dict)
    except IOError as e:
        log.debug("Cannot read skin configuration file %s for report '%s': %s",
                  skin_config_path, report, e)
    except SyntaxError as e:
        log.error("Failed to read skin configuration file %s for report '%s': %s",
                  skin_config_path, report, e)
        raise

    # Now add on the [StdReport][[Defaults]] section, if present:
    if 'Defaults' in config_dict['StdReport']:
        # Because we will be modifying the results, make a deep copy of the [[Defaults]]
        # section.
        merge_dict = weeutil.config.deep_copy(config_dict)['StdReport']['Defaults']
        weeutil.config.merge_config(skin_dict, merge_dict)

    # Inject any scalar overrides. This is for backwards compatibility. These options should now go
    # under [StdReport][[Defaults]].
    for scalar in config_dict['StdReport'].scalars:
        skin_dict[scalar] = config_dict['StdReport'][scalar]

    # Finally, inject any overrides for this specific report. Because this is the last merge, it will have the
    # final say.
    weeutil.config.merge_config(skin_dict, config_dict['StdReport'][report])

    return skin_dict

if __name__ == '__main__':
    import configobj
    import optparse
    import socket
    parser = optparse.OptionParser()
    parser.add_option("--config", dest="config_path", metavar="CONFIG_FILE",
                      default="/home/weewx/weewx.conf",
                      help="Use configuration file CONFIG_FILE")
    parser.add_option("--debug", action="store_true",
                      help="Enable verbose logging")
    (options, args) = parser.parse_args()

    debug = 0
    if options.debug:
        weewx.debug = 2
        debug = 1

    cfg = dict()
    try:
        cfg = configobj.ConfigObj(options.config_path)
    except IOError:
        pass

    stdrep = cfg.get('StdReport', {})
    gen = PolarPlotGenerator(
        cfg, build_skin_dict(cfg, 'SeasonsReport') , gen_ts=None, first_run=None, stn_info=weewx.station.StationInfo(**cfg['Station']))
    gen.run()
