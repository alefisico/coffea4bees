class Plot:
    figure_height = 400  # px, height of the figure
    ratio_height = figure_height / 2  # px, height of the ratio plot
    fill_alpha = 0.2  # transparency of the fill color, 0-1
    legend_width = 200  # px, width of the legend
    tooltip_float_precision = 4  # float precision for tooltips
    # constants
    legend_checkbox_width = 27  # px, width of the bokeh checkbox
    legend_glyph_width = legend_checkbox_width * 3  # px
    legend_disabled_color = "#E5E5E5"  # color of disabled legend items


class UI:
    path_separator = "."  # path separator for hist names
    # layout
    log_height = 50  # px, height of the log
    sidebar_width = 200  # px, width of the treeview and category selector
    # widgets
    multichoice_height = 40  # px, height of the multichoice widget
    numeric_input_width = 80  # px, width of the numeric input widget
    # colors
    background_color = "#E8E8E8"  # color of the background
    border_color = "#C8C8C8"  # color of the border
    border = f"1px solid {border_color}"  # border style
    badge_color = {
        "model": "#385CB4",
        "stack": "#29855A",
        "ratio": "#E02D4B",
    }
