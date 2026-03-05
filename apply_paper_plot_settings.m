function style = apply_paper_plot_settings()
    style = struct();

    % Global defaults for paper-ready, multi-panel readability.
    style.axesFontSize = 16;
    style.labelFontSize = 20;
    style.legendFontSize = 14;
    style.titleFontSize = 18;
    style.lineWidth = 2.0;
    style.highlightLineWidth = 2.5;
    style.markerSize = 7;
    style.flutterMarkerSize = 16;
    style.scatterSizeMedium = 80;
    style.scatterSizeLarge = 260;
    style.figureColor = 'w';
    style.figurePosition = [100, 100, 1800, 900];
    style.tileSpacing = 'compact';
    style.tilePadding = 'compact';

    set(groot, ...
        'defaultAxesFontSize', style.axesFontSize, ...
        'defaultAxesTickLabelInterpreter', 'latex', ...
        'defaultTextInterpreter', 'latex', ...
        'defaultLegendInterpreter', 'latex', ...
        'defaultLineLineWidth', style.lineWidth, ...
        'defaultAxesLineWidth', 1.1);
end
