import os
import pandas as pd
import numpy as np
from collections import OrderedDict

from pipeline.steps import PipelineStep
from externals.gists.mclust.gaussian_mixture import Mclust
from pipeline.steps.visualization.classification_effect_size_plot_matplotlib import \
    ClassificationEffectSizePlotMatplotlib

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable

# set font
font_dir = "./fonts/"
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)
plt.rcParams['font.family'] = 'Share Tech'


class SummaryPlotMatplotlib(PipelineStep):
    def _execute(self, group: list, result_folder_analyses: list,
                 modality_names: list = None, corrected_p: bool = True,
                 plot_p_values: bool = True,
                 use_sample_filter: bool = True,
                 *args, **kwargs):
        result_folder_analyses = zip(group, result_folder_analyses)
        self.result_folder = result_folder_analyses
        self.corrected_p = corrected_p
        self.plot_p_values = plot_p_values

        mod_names = list()
        mods = pd.DataFrame()
        residuals = {}
        width = list()
        group = list()
        sections = OrderedDict()

        for i, folder in enumerate(self.result_folder):

            if not use_sample_filter:
                base_folder, filter_name = os.path.split(folder[1])
                folder = (folder[0], base_folder)
            else:
                filter_name = self.pipeline.filter_name.split('filter_')[1]

            group.append(folder[0])
            width.append(0.7)
            if modality_names:
                name = modality_names[i]
            else:
                name = os.path.split(folder[1])[-1]
            mod_names.append(name)

            if folder[0] in sections.keys():
                sections[folder[0]].append(name)
            else:
                sections[folder[0]] = [name]

            analysis_folder = os.path.join(folder[1], filter_name)

            results = pd.read_csv(os.path.join(analysis_folder, 'effect_size_results.csv'))
            results['Unnamed: 0'] = i
            aov = pd.read_csv(os.path.join(analysis_folder, 'anova_results.csv'))
            if self.corrected_p:
                if hasattr(aov, 'p-corr'):
                    results['p'] = aov['p-corr'][1]
                else:
                    p = pd.read_csv(os.path.join(analysis_folder, 'spm_p.csv'))
                    results['p'] = p['p-corr']
                results['p-corr'] = results['p']
            else:
                results['p'] = aov['p-unc'][1]

            results['p-unc'] = aov['p-unc'][1]
            results['Partial Eta2 Upper'] = aov['np2_BCI_high'][1]
            results['Partial Eta2 Lower'] = aov['np2_BCI_low'][1]
            results['F'] = aov['F'][1]
            results['DF_1'] = aov['DF'][1]
            n_predictors = aov.shape[0]
            results['DF_2'] = aov['DF'][n_predictors - 1]
            # n of the specific analyses can be computed by the error degrees of freedom
            # plus the number of predictors (which is the number of rows of the anova result
            # table minus 2 to substract the row for the residuals and the intercept) plus 1
            # for the group degrees of freedom (which is always 1 in this case)
            results['n'] = results['DF_2'] + n_predictors - 1
            mods = pd.concat([mods, results])

            resid = pd.read_csv(os.path.join(analysis_folder, 'residuals.csv'))
            resid = resid.sort_values(by=['Group'], ascending=True)
            residuals[name] = resid

        mods['name'] = mod_names

        # save information of all modalities to csv file
        mods.to_csv(os.path.join(self.result_dir, 'effect_size_and_accuracy.csv'))

        n_mods = mods.shape[0]
        self.n_mods = n_mods
        self.mods = mods
        self.mod_names = mod_names
        self.residuals = residuals
        self.width = width
        self.group = group
        self.sections = sections
        self._plot_effect_size_matplotlib()
        plt.savefig(os.path.join(self.result_dir, 'effect_size_plot_matplotlib.png'))
        self._plot_accuracy_matplotlib()
        plt.savefig(os.path.join(self.result_dir, 'accuracy_plot_matplotlib.png'))
        plt.show()

    def _plot_effect_size_matplotlib(self):
        n_bars = self.n_mods
        sections = self.sections

        # get colors and y coords
        n_sections = len(sections.keys())
        colors = sns.color_palette('colorblind', n_sections)

        colors_bars = list()
        y_coord_bars = list()
        y_coord_sections = list()
        y_coord = 0
        names = list()
        for section_i, section_bars in enumerate(sections.values()):
            y_coord_sections.append(y_coord)
            y_coord -= 0.25

            for bar_name in section_bars:
                names.append(bar_name)
                y_coord -= 0.5
                y_coord_bars.append(y_coord)
                y_coord -= 0.5
                colors_bars.append(colors[section_i])
            y_coord -= 0.25

        m = self.mods['Partial Eta2'].values
        upper = self.mods['Partial Eta2 Upper'].values
        lower = self.mods['Partial Eta2 Lower'].values

        # start drawing the figure
        # todo: check if we really need GridSpec here
        fig = plt.figure(constrained_layout=True, figsize=(5, 3), dpi=400)
        gs = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
        ax = fig.add_subplot(gs[:, :])

        # plot error bars
        for bar_i in range(n_bars):
            y_coord = y_coord_bars[bar_i]
            #ax.plot([lower[bar_i], upper[bar_i]], [y_coord, y_coord], color=colors_bars[bar_i], alpha=0.35, linewidth=1,
            #        solid_capstyle='round')
            ax.plot([lower[bar_i], upper[bar_i]], [y_coord, y_coord], color='k', alpha=0.5, linewidth=1,
                    solid_capstyle='round')
            ax.plot([0, m[bar_i]], [y_coord, y_coord], color=colors_bars[bar_i], alpha=0.5, linewidth=4,
                    solid_capstyle='round')

        xmin = 0
        xmax = 0.5
        y_min = y_coord_bars[-1] - 0.5

        # plot grey shadow for bars
        for bar_i in range(n_bars):
            start = y_coord_bars[bar_i] + 0.48
            end = y_coord_bars[bar_i] - 0.48
            ax.fill_between(
                [xmin, xmax],
                [end, end],
                [start, start],
                color='#939596', alpha=0.15, linewidth=0)

        # plot lines for small, medium, large effect
        ax.vlines([0.09, 0.09], y_min + 0.02, 0.23, color='#DCDCDC', linestyle='-', lw=1)
        ax.vlines([0.25, 0.25], y_min + 0.02, 0.23, color='#DCDCDC', linestyle='-', lw=1)
        ax.vlines([0.40, 0.40], y_min + 0.02, 0.23, color='#DCDCDC', linestyle='-', lw=1)
        ax.text(0.045, 0.35, "small", fontsize=5, horizontalalignment='center')
        ax.text(0.17, 0.35, "medium", fontsize=5, horizontalalignment='center')
        ax.text(0.325, 0.35, "large", fontsize=5, horizontalalignment='center')

        # plot p values
        if self.plot_p_values:
            for i in range(n_bars):
                p_value = self.mods['p'].tolist()[i]
                if p_value < 0.001:
                    p_txt = r'$p < 0.001$ ***'
                elif p_value < 0.01:
                    p_txt = r'$p = {:.3f}$ **'.format(p_value)
                elif p_value < 0.05:
                    p_txt = r'$p = {:.3f}$ *'.format(p_value)
                else:
                    p_txt = r'$p = {:.3f}$'.format(p_value)
                ax.text(upper[i] + 0.02, y_coord_bars[i], p_txt, fontsize=3,
                        verticalalignment='center')

        # plot grey shadow for section dividers
        for section_i in range(n_sections):
            start = y_coord_sections[section_i] + 0.23
            end = y_coord_sections[section_i] - 0.23
            ax.fill_between(
                [xmin, xmax],
                [end, end],
                [start, start],
                color='#D2D2D2', alpha=1, linewidth=0, zorder=10)

        ax.set_yticks(ticks=y_coord_bars)
        ax.set_xticks(ticks=[0, 0.09, 0.25, 0.4])
        plt.xlabel(r'partial $\eta^2$', fontsize=5)
        ax.tick_params(axis='x', labelsize=5)

        # style
        ax.set_ylim([y_min, 0.25])
        ax.set_xlim([xmin, xmax])
        ax.spines['bottom'].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        ax.xaxis.set_tick_params(pad=-3)

        # plot additional axis to get fancy y tick labels
        divider = make_axes_locatable(ax)
        # 50% might be adjusted depending on the length of the y tick labels
        cax = divider.append_axes("left", size="30%", pad=0)

        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.spines['bottom'].set_visible(False)
        cax.spines["right"].set_visible(False)
        cax.spines["left"].set_visible(False)
        cax.spines["top"].set_visible(False)
        cax.set_ylim([y_min, 0.25])
        cax.set_xlim([xmin, xmax])

        # plot grey shadow for bars
        for bar_i in range(n_bars):
            start = y_coord_bars[bar_i] + 0.48
            end = y_coord_bars[bar_i] - 0.48
            cax.fill_between(
                [xmin, xmax],
                [end, end],
                [start, start],
                color=colors_bars[bar_i], alpha=0.15, linewidth=0)
            cax.text(xmax - 0.05 * (xmax - xmin), y_coord_bars[bar_i], names[bar_i],
                     verticalalignment='center', horizontalalignment='right',
                     fontsize=6)

        # plot grey shadow for section dividers
        section_names = list(sections.keys())
        for section_i in range(n_sections):
            start = y_coord_sections[section_i] + 0.23
            end = y_coord_sections[section_i] - 0.23
            cax.fill_between(
                [xmin, xmax],
                [end, end],
                [start, start],
                color=colors[section_i], alpha=0.25, linewidth=0)
            cax.text(xmin + xmax * 0.05, y_coord_sections[section_i], section_names[section_i],
                     verticalalignment='center', horizontalalignment='left',
                     fontsize=4.5)

        return fig, ax, cax

    def _plot_accuracy_matplotlib(self):
        n_bars = self.n_mods
        sections = self.sections

        # get colors and y coords
        n_sections = len(sections.keys())
        colors = sns.color_palette('colorblind', n_sections)

        colors_bars = list()
        y_coord_bars = list()
        y_coord_sections = list()
        y_coord = 0
        height_ratios = list()
        names = list()
        for section_i, section_bars in enumerate(sections.values()):
            y_coord_sections.append(y_coord)
            height_ratios.append(0.5)
            y_coord -= 0.25

            for bar_name in section_bars:
                height_ratios.append(1)
                names.append(bar_name)
                y_coord -= 0.5
                y_coord_bars.append(y_coord)
                y_coord -= 0.5
                colors_bars.append(colors[section_i])
            y_coord -= 0.25

        m = self.mods['BalancedAccuracy'].values

        # start drawing the figure
        # todo: check if we really need GridSpec here
        fig = plt.figure(constrained_layout=False, figsize=(7, 3), dpi=400)
        gs = gridspec.GridSpec(ncols=8, nrows=n_bars + n_sections, figure=fig,
                               height_ratios=height_ratios, hspace=0, wspace=0.02)
        ax = fig.add_subplot(gs[:, 4:-1])

        # plot error bars
        for bar_i in range(n_bars):
            y_coord = y_coord_bars[bar_i]
            ax.plot([0, m[bar_i]], [y_coord, y_coord], color=colors_bars[bar_i], alpha=0.55, linewidth=4,
                    solid_capstyle='round')

        xmin = 0.5
        xmax = 1
        y_min = y_coord_bars[-1] - 0.5

        # plot grey shadow for bars
        for bar_i in range(n_bars):
            start = y_coord_bars[bar_i] + 0.48
            end = y_coord_bars[bar_i] - 0.48
            ax.fill_between(
                [xmin, xmax],
                [end, end],
                [start, start],
                color='#F0F0F0', alpha=1, linewidth=0)

        # plot lines for small, medium, large effect
        ax.vlines([0.6, 0.6], y_min + 0.02, 0.23, color='#DCDCDC', linestyle='-', lw=1)
        ax.vlines([0.7, 0.7], y_min + 0.02, 0.23, color='#DCDCDC', linestyle='-', lw=1)
        ax.vlines([0.8, 0.8], y_min + 0.02, 0.23, color='#DCDCDC', linestyle='-', lw=1)
        ax.vlines([0.9, 0.9], y_min + 0.02, 0.23, color='#DCDCDC', linestyle='-', lw=1)

        # plot grey shadow for section dividers
        for section_i in range(n_sections):
            start = y_coord_sections[section_i] + 0.23
            end = y_coord_sections[section_i] - 0.23
            ax.fill_between(
                [xmin, xmax],
                [end, end],
                [start, start],
                color='#D2D2D2', alpha=1, linewidth=0, zorder=10)

        ax.set_yticks(ticks=y_coord_bars)
        ax.set_xticks(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plt.xlabel('balanced accuracy', fontsize=6)
        ax.tick_params(axis='x', labelsize=6)

        # style
        ax.set_ylim([y_min, 0.25])
        ax.set_xlim([xmin, xmax])
        ax.spines['bottom'].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        ax.xaxis.set_tick_params(pad=-3)

        """
        create effect size plot
        """

        # plot effect size
        # plot white divider line between rows
        #divider = make_axes_locatable(ax)
        # 50% might be adjusted depending on the length of the y tick labels
        #dax = divider.append_axes("right", size="30%", pad=0)
        dax = fig.add_subplot(gs[:, 0:4])
        # plot error bars
        # plot error bars
        m = self.mods['Partial Eta2'].values
        upper = self.mods['Partial Eta2 Upper'].values
        lower = self.mods['Partial Eta2 Lower'].values

        xmin = 0
        xmax = 0.402
        y_min = y_coord_bars[-1] - 0.5

        # plot grey shadow for bars
        for bar_i in range(n_bars):
            start = y_coord_bars[bar_i] + 0.48
            end = y_coord_bars[bar_i] - 0.48
            dax.fill_between(
                [xmin, xmax],
                [end, end],
                [start, start],
                color='#939596', alpha=0.15, linewidth=0)

        # plot lines for small, medium, large effect
        dax.vlines([0.09, 0.09], y_min + 0.02, 0.23, color='#DCDCDC', linestyle='-', lw=1)
        dax.vlines([0.25, 0.25], y_min + 0.02, 0.23, color='#DCDCDC', linestyle='-', lw=1)
        dax.vlines([0.40, 0.40], y_min + 0.02, 0.23, color='#DCDCDC', linestyle='-', lw=1)
        dax.text(0.045, 0.35, "small", fontsize=5, horizontalalignment='center')
        dax.text(0.17, 0.35, "medium", fontsize=5, horizontalalignment='center')
        dax.text(0.325, 0.35, "large", fontsize=5, horizontalalignment='center')



        dax.set_yticks(ticks=y_coord_bars)
        dax.set_xticks(ticks=[0, 0.09, 0.25, 0.4])
        plt.xlabel(r'partial $\eta^2$', fontsize=6)
        dax.tick_params(axis='x', labelsize=6)


        # plot grey shadow for bars
        for bar_i in range(n_bars):
            start = y_coord_bars[bar_i] + 0.48
            end = y_coord_bars[bar_i] - 0.48
            dax.fill_between(
                [xmin, xmax],
                [end, end],
                [start, start],
                color='#F0F0F0', alpha=1, linewidth=0)

        # plot grey shadow for section dividers
        for section_i in range(n_sections):
            start = y_coord_sections[section_i] + 0.23
            end = y_coord_sections[section_i] - 0.23
            dax.fill_between(
                [xmin, xmax],
                [end, end],
                [start, start],
                color='#D2D2D2', alpha=1, linewidth=0, zorder=10)

        for bar_i in range(n_bars):
            y_coord = y_coord_bars[bar_i]
            # ax.plot([lower[bar_i], upper[bar_i]], [y_coord, y_coord], color=colors_bars[bar_i], alpha=0.35, linewidth=1,
            #        solid_capstyle='round')
            dax.plot([lower[bar_i], upper[bar_i]], [y_coord, y_coord], color='k', alpha=0.5, linewidth=1,
                    solid_capstyle='round')
            dax.plot([0, m[bar_i]], [y_coord, y_coord], color=colors_bars[bar_i], alpha=0.5, linewidth=4,
                    solid_capstyle='round')

        # style
        dax.set_ylim([y_min, 0.25])
        dax.set_xlim([xmin, xmax])

        dax.spines['bottom'].set_visible(False)
        dax.spines["right"].set_visible(False)
        dax.spines["left"].set_visible(False)
        dax.spines["top"].set_visible(False)
        dax.yaxis.set_ticks_position('none')
        dax.xaxis.set_ticks_position('none')
        dax.xaxis.set_tick_params(pad=-3)
        dax.get_yaxis().set_visible(False)
        plt.gca().invert_xaxis()


        """
        create labels
        """
        # plot additional axis to get fancy y tick labels
        divider = make_axes_locatable(dax)
        # 50% might be adjusted depending on the length of the y tick labels
        cax = divider.append_axes("right", size="50%", pad=0.02)

        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.spines['bottom'].set_visible(False)
        cax.spines["right"].set_visible(False)
        cax.spines["left"].set_visible(False)
        cax.spines["top"].set_visible(False)
        cax.set_ylim([y_min, 0.25])
        cax.set_xlim([xmin, xmax])

        # plot grey shadow for bars
        for bar_i in range(n_bars):
            start = y_coord_bars[bar_i] + 0.48
            end = y_coord_bars[bar_i] - 0.48
            cax.fill_between(
                [xmin, xmax],
                [end, end],
                [start, start],
                color=colors_bars[bar_i], alpha=0.15, linewidth=0)
            cax.text(xmin + (xmax - xmin) / 2, y_coord_bars[bar_i], names[bar_i],
                     verticalalignment='center', horizontalalignment='center',
                     fontsize=6)

        # plot grey shadow for section dividers
        section_names = list(sections.keys())
        for section_i in range(n_sections):
            start = y_coord_sections[section_i] + 0.23
            end = y_coord_sections[section_i] - 0.23
            cax.fill_between(
                [xmin, xmax],
                [end, end],
                [start, start],
                color=colors[section_i], alpha=0.25, linewidth=0)
            cax.text(xmin + (xmax - xmin) / 2, y_coord_sections[section_i], section_names[section_i],
                     verticalalignment='center', horizontalalignment='center',
                     fontsize=4.5)

        # plot kde plots
        cnt = 0
        bar_i = 0
        for section_i, section_bars in enumerate(sections.values()):
            kdeax = fig.add_subplot(gs[cnt, 7])
            kdeax.get_xaxis().set_visible(False)
            kdeax.get_yaxis().set_visible(False)
            kdeax.spines['bottom'].set_visible(False)
            kdeax.spines["right"].set_visible(False)
            kdeax.spines["left"].set_visible(False)
            kdeax.spines["top"].set_visible(False)
            kdeax.set_facecolor('#D2D2D2')

            cnt += 1
            for bar_name in section_bars:
                df = self.residuals[self.mod_names[bar_i]]
                residuals = df['residuals']
                group_df = df['Group']
                unique_values = group_df.unique()
                unique_values = unique_values[::-1]
                x1 = residuals[group_df == unique_values[0]].values
                x2 = residuals[group_df == unique_values[1]].values

                overlap = self._calculate_overlap(x1, x2)

                kdeax = fig.add_subplot(gs[cnt, 7])

                kdeax.text(0.7, 0.65, "{:.1f}%".format(overlap * 100), fontsize=5,
                           verticalalignment='center', transform=kdeax.transAxes)

                sns.kdeplot(data=df, x="residuals", hue="Group",
                            fill=True, common_norm=False,
                            alpha=.5, linewidth=0, ax=kdeax, legend=False,
                            hue_order=['MDD', 'HC'])

                kdeax.set_facecolor('#F0F0F0')
                if cnt == 1:
                    # super strange thing happening here
                    # somehow the legend added afterwards changes the order of HC and MDD
                    # that's why I used this pretty annoying workaround and manually set the color
                    # for MDD and HC, which should now correspond to the original kde plot colors
                    import matplotlib.patches as mpatches
                    c1 = sns.color_palette()[0]
                    c1 = (c1[0], c1[1], c1[2], 0.5)
                    c2 = sns.color_palette()[1]
                    c2 = (c2[0], c2[1], c2[2], 0.5)
                    handles = [mpatches.Patch(facecolor=c1, label="MDD"),
                               mpatches.Patch(facecolor=c2, label="HC")]
                    kdeax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.6, 1.5), prop={'size': 5},
                                 frameon=False)

                kdeax.get_xaxis().set_visible(False)
                kdeax.get_yaxis().set_visible(False)
                kdeax.spines['bottom'].set_visible(False)
                kdeax.spines["right"].set_visible(False)
                kdeax.spines["left"].set_visible(False)
                kdeax.spines["top"].set_visible(False)
                # Turn off tick labels

                kdeax.yaxis.set_ticks_position('none')
                kdeax.xaxis.set_ticks_position('none')

                bar_i += 1
                cnt += 1
        kdeax.get_xaxis().set_visible(True)
        kdeax.set_xlabel('overlap', labelpad=2, fontsize=6)
        kdeax.set_xticklabels([])

        # plot white divider line between rows
        ax = fig.add_subplot(gs[:, :], label='divider_axis')
        ax.patch.set_alpha(0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_ylim([y_min, 0.25])
        ax.set_xlim([xmin, xmax])

        for section_i in range(n_sections):
            start = y_coord_sections[section_i] + 0.23
            end = y_coord_sections[section_i] + 0.25
            ax.fill_between(
                [xmin, xmax],
                [end, end],
                [start, start],
                color='w', alpha=1, linewidth=0)

            start = y_coord_sections[section_i] - 0.23
            end = y_coord_sections[section_i] - 0.25
            ax.fill_between(
                [xmin, xmax],
                [end, end],
                [start, start],
                color='w', alpha=1, linewidth=0)

        # plot white divider for bars
        for bar_i in range(n_bars):
            start = y_coord_bars[bar_i] + 0.48
            end = y_coord_bars[bar_i] + 0.5
            ax.fill_between(
                [xmin, xmax],
                [end, end],
                [start, start],
                color='w', alpha=1, linewidth=0)
            start = y_coord_bars[bar_i] - 0.48
            end = y_coord_bars[bar_i] - 0.5
            ax.fill_between(
                [xmin, xmax],
                [end, end],
                [start, start],
                color='w', alpha=1, linewidth=0)

        return fig, ax, cax

    @staticmethod
    def _calculate_overlap(data1, data2):
        x1 = data1
        x2 = data2
        mclust1 = Mclust("V", n_gaussians=1)
        mclust1.fit(x1)
        mu1 = np.asarray(mclust1.model.rx2['parameters'].rx2['mean'])[0]
        sigma1 = np.sqrt(np.asarray(mclust1.model.rx2['parameters'].rx2['variance'].rx2['sigmasq']))
        mclust2 = Mclust("V", n_gaussians=1)
        mclust2.fit(x2)
        mu2 = np.asarray(mclust2.model.rx2['parameters'].rx2['mean'])[0]
        sigma2 = np.sqrt(np.asarray(mclust2.model.rx2['parameters'].rx2['variance'].rx2['sigmasq']))

        return ClassificationEffectSizePlotMatplotlib._normdist_calculate_overlap(mu1, sigma1, mu2, sigma2)[0]
