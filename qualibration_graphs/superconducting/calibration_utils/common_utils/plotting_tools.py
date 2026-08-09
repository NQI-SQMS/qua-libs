import matplotlib.pyplot as plt


def patch_fig_info(node, more_info=""):
    snapshot_idx = node.snapshot_idx

    for num in plt.get_fignums():
        fig = plt.figure(num)

        st = fig._suptitle.get_text() if fig._suptitle else ""
        st += "\n" + more_info
        have_st = fig._suptitle is not None

        if have_st and not st.lstrip().startswith("#"):
            # prepend to existing
            fs = fig._suptitle.get_fontsize()
            pos = fig._suptitle.get_position()
            align = fig._suptitle.get_ha()
            fig.suptitle(f"#{snapshot_idx} - " + st, fontsize=fs, x=pos[0], y=pos[1], horizontalalignment=align)
        elif not have_st:
            # create new suptitle
            fig.suptitle(f"#{snapshot_idx}", fontsize=32)
            fig.subplots_adjust(top=0.80)  # make room only when adding fresh
