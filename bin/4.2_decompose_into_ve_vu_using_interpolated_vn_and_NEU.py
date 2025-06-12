#!/usr/bin/env python3
import os
from merge_tif import *
from modules import *
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s -- %(levelname)s -- %(message)s')
logger = logging.getLogger('decomposition.log')


class Layer:
    def __init__(self, track_list, data_dir, sig_dir, neu_dir, data_suffix, sig_suffix, n_suffix, e_suffix, u_suffix):
        self.gp = {}

        # define layer boundary
        left = []
        right = []
        top = []
        bottom = []
        xres = []
        yres = []

        for trk in track_list:
            logger.info("loading data from {}".format(trk))
            data_path = os.path.join(data_dir, trk, trk + data_suffix)
            if not os.path.exists(data_path):
                logger.warning(f"Skipping {trk} — LOS file not found: {data_path}")
                continue
            self.gp[trk] = OpenTif(data_dir + trk + '/' + trk + data_suffix,
                                   sigfile=sig_dir + trk + '/' + trk + sig_suffix,
                                   N=neu_dir + trk + '/' + trk + n_suffix,
                                   E=neu_dir + trk + '/' + trk + e_suffix,
                                   U=neu_dir + trk + '/' + trk + u_suffix)
            
            # After collecting xres and yres
            xres = [round(val, 5) for val in xres]
            yres = [round(val, 5) for val in yres]
            # keep track of boundary
            left.append(self.gp[trk].left)
            right.append(self.gp[trk].right)
            top.append(self.gp[trk].top)
            bottom.append(self.gp[trk].bottom)
            xres.append(self.gp[trk].xres)
            yres.append(self.gp[trk].yres)
        # print(xres, yres)
        if len(set(xres)) > 1 or len(set(yres)) > 1:
            raise Warning("tracks have different resolution")

        # define the geographic boundary of the map
        self.top = max(top)
        self.bottom = min(bottom)
        self.left = min(left)
        self.right = max(right)
        self.xres = xres[0]
        self.yres = yres[0]

        self.data = None
        self.sig = None
        self.N = None
        self.E = None
        self.U = None

    def place_data_sig_N_E_U_to_canvas(self, maps, plotting=False, layer_name="layer", output_dir="canvas_layer"):
        # logger.info("Placing data_sig_N_E_U into canvas")
        self.data = np.ones((maps.ysize, maps.xsize)) * np.nan
        self.sig = np.ones((maps.ysize, maps.xsize)) * np.nan
        self.N = np.ones((maps.ysize, maps.xsize)) * np.nan
        self.E = np.ones((maps.ysize, maps.xsize)) * np.nan
        self.U = np.ones((maps.ysize, maps.xsize)) * np.nan
        for i in self.gp.keys():
            # print(i)
            x_shift = int((self.gp[i].left - maps.left) / maps.xres + 0.5)
            y_shift = int((self.gp[i].top - maps.top) / maps.yres + 0.5)
            nodata_test = np.isnan(layer.gp[i].data)  # = True if nan, = False if not nan; True = 1, False = 0
            
            non_nan_merge(self.data, self.gp[i].data, nodata_test, x_shift, y_shift, self.gp[i].xsize, self.gp[i].ysize)
            non_nan_merge(self.sig, self.gp[i].sigma, nodata_test, x_shift, y_shift, self.gp[i].xsize, self.gp[i].ysize)
            non_nan_merge(self.N, self.gp[i].N, nodata_test, x_shift, y_shift, self.gp[i].xsize, self.gp[i].ysize)
            non_nan_merge(self.E, self.gp[i].E, nodata_test, x_shift, y_shift, self.gp[i].xsize, self.gp[i].ysize)
            non_nan_merge(self.U, self.gp[i].U, nodata_test, x_shift, y_shift, self.gp[i].xsize, self.gp[i].ysize)

        if plotting:
            fig, ax = plt.subplots()  # Create a single figure and axis for the combined data
            im = ax.imshow(self.data, vmin=np.nanpercentile(self.data, 10), vmax=np.nanpercentile(self.data, 90),
                                 interpolation="nearest", cmap='RdBu_r')
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("right", size="7%", pad="5%")
            fig.colorbar(im, cax=cax)
            ax.set_title("data")
            
            output_file = os.path.join(output_dir, f'plot_{layer_name}.png') # Modify 'self.output_dir' to your desired path
            plt.savefig(output_file)
            plt.close()  # Close the figure to free up memory

    def flatten_and_mask(self, mask_test):
        self.data = self.data.flatten()[mask_test]
        self.sig = self.sig.flatten()[mask_test]
        self.N = self.N.flatten()[mask_test]
        self.E = self.E.flatten()[mask_test]
        self.U = self.U.flatten()[mask_test]


def define_canvas_size(layers):
    """ Define size of the big raster grid, big enough to host all Vn and LOS tracks."""
    left = []
    right = []
    top = []
    bottom = []
    xres = []
    yres = []
    
    for layer in layers:
        left.append(layer.left)
        right.append(layer.right)
        top.append(layer.top)
        bottom.append(layer.bottom)
        xres.append(layer.xres)
        yres.append(layer.yres)
    print(xres, yres)

    if len(set(xres)) > 1 or len(set(yres)) > 1:
        print(xres)
        print(yres)
        raise Warning("layers have different resolution")

    # define the geographic boundary of the map
    logger.info("define the geographic boundary of the canvas")
    canvas = Canvas(north=max(top), south=min(bottom), west=min(left), east=max(right), x_step=xres[0], y_step=yres[0])
    return canvas


def must_have(group_list, at_least=1):
    "list_of_group_lists = must have all of [[must have one], [must have one], [must have one]]"
    must_have = np.zeros(group_list[0].data.flatten().size)
    for gp in group_list:
        must_have += ~np.isnan(gp.data.flatten())
    return must_have >= at_least


def decomposition_inversion(layers):
    """ layers = a list of Layer """
    # empty arrays to register inversion results
    length = layers[0].data.size
    vn_mask = np.ones(length)
    ve_mask = np.ones(length)
    vu_mask = np.ones(length)
    vn_sig_mask = np.ones(length)
    ve_sig_mask = np.ones(length)
    vu_sig_mask = np.ones(length)
    
    logger.info(f"3D inversion starts... total {length} pixels")
    
    # LOS = Vn * N + Ve * E + Vu * U
    for c in np.arange(length):
        d = np.ones((len(layers), 1)) * np.nan
        sig = np.ones((len(layers), 1)) * np.nan
        G = np.ones((len(layers), 3)) * np.nan
        for i, ly in enumerate(layers):
            d[i] = ly.data[c]
            sig[i] = ly.sig[c]
            G[i] = [ly.N[c], ly.E[c], ly.U[c]]
        # drop nan rows
        nonnan = np.logical_and(~np.isnan(d).any(axis=1), ~np.isnan(G).any(axis=1))
        sig = sig[nonnan]
        d = d[nonnan]
        G = G[nonnan]

        try:
            [[n], [e], [u]] = np.linalg.lstsq(G / sig, d / sig, rcond=None)[0]
            cov_d = np.diag(np.square(sig).transpose()[0])
            cov_m = np.linalg.inv(np.dot(np.dot(G.transpose(), np.linalg.inv(cov_d)), G))
            if c % (length // 100) == 0:
                logger.info("{}, {:.2f}%, {:.2f}, {:.2f}".format(c, c / length * 100, e, u))
            vn_mask[c] = n
            ve_mask[c] = e
            vu_mask[c] = u
            vn_sig_mask[c] = np.sqrt(cov_m[0, 0])
            ve_sig_mask[c] = np.sqrt(cov_m[1, 1])
            vu_sig_mask[c] = np.sqrt(cov_m[2, 2])
        except:
            vn_mask[c] = np.nan
            ve_mask[c] = np.nan
            vu_mask[c] = np.nan
            vn_sig_mask[c] = np.nan
            ve_sig_mask[c] = np.nan
            vu_sig_mask[c] = np.nan
    return vn_mask, ve_mask, vu_mask, vn_sig_mask, ve_sig_mask, vu_sig_mask

def decomposition_inversion_eu(layers):
    """
    Invert LOS = Ve * E + Vu * U to get Ve, Vu and their uncertainties.
    `layers` should be a list of Layer objects with .data, .sig, .E, .U defined and flattened.
    """
    length = layers[0].data.size
    ve_mask = np.full(length, np.nan)
    vu_mask = np.full(length, np.nan)
    ve_sig_mask = np.full(length, np.nan)
    vu_sig_mask = np.full(length, np.nan)

    logger.info(f"2D inversion (Ve, Vu only) starts... total {length} pixels")

    for c in range(length):
        d = []
        sig = []
        G = []

        for ly in layers:
            d_val = ly.data[c]
            sig_val = ly.sig[c]
            e_val = ly.E[c]
            u_val = ly.U[c]

            if not np.isnan(d_val) and not np.isnan(sig_val) and not np.isnan(e_val) and not np.isnan(u_val):
                d.append([d_val])
                sig.append([sig_val if sig_val > 0 else 1.0])  # avoid divide by zero
                G.append([e_val, u_val])

        if len(d) < 2:
            continue  # not enough data to solve 2 parameters

        d = np.array(d)
        sig = np.array(sig)
        G = np.array(G)

        try:
            [[e], [u]] = np.linalg.lstsq(G / sig, d / sig, rcond=None)[0]

            cov_d = np.diagflat(sig**2)
            GT_Cinv_G = G.T @ np.linalg.inv(cov_d) @ G
            cov_m = np.linalg.inv(GT_Cinv_G)

            ve_mask[c] = e
            vu_mask[c] = u
            ve_sig_mask[c] = np.sqrt(cov_m[0, 0]) if cov_m[0, 0] >= 0 else np.nan
            vu_sig_mask[c] = np.sqrt(cov_m[1, 1]) if cov_m[1, 1] >= 0 else np.nan

            if c % (length // 100) == 0:
                logger.info(f"{c}, {c / length * 100:.2f}%, Ve = {e:.2f}, Vu = {u:.2f}")

        except Exception as ex:
            logger.warning(f"Inversion failed at pixel {c}: {ex}")
            continue

    logger.info("Done with pixel-wise 2D (Ve, Vu) inversion.")
    return ve_mask, vu_mask, ve_sig_mask, vu_sig_mask



def remove_vn_contribution_from_los_and_propagate_sigma_n_to_sigma_los(layer, vn):
    for trk in layer.gp:
        print(trk)
        vn_overlap = Overlap(layer.gp[trk], vn)
        layer.gp[trk].data -= vn_overlap.d2array * layer.gp[trk].N
        layer.gp[trk].sigma = np.sqrt(
            np.square(layer.gp[trk].sigma) + np.square(vn_overlap.d2sigma * layer.gp[trk].N)
        )

def back_2d(arr_1d, mask_test, canvas):
    arr = np.ones(mask_test.size) * np.nan
    arr[mask_test] = arr_1d
    return arr.reshape((canvas.ysize, canvas.xsize))


def reference(data, canvas, lat, lon, window_pixel_size):
    ref_x = int((lon - canvas.left) / canvas.xres + 0.5)
    ref_y = int((lat - canvas.top) / canvas.yres + 0.5)
    ref_x0 = ref_x - window_pixel_size
    ref_y0 = ref_y - window_pixel_size
    ref_x1 = ref_x + window_pixel_size
    ref_y1 = ref_y + window_pixel_size
    
    # Ensure we stay within array bounds
    ref_x0 = max(0, ref_x0)
    ref_y0 = max(0, ref_y0)
    ref_x1 = min(canvas.xsize, ref_x1)
    ref_y1 = min(canvas.ysize, ref_y1)
    
    # Calculate median value in the window and subtract it from the data
    median_value = np.nanmedian(data[ref_y0:ref_y1, ref_x0:ref_x1])
    data -= median_value
    
    return ref_x0, ref_x1, ref_y0, ref_y1


if __name__ == "__main__":
    #################################################
    # Define input directories and file suffix here.#
    # #################################################
    # base_dir = "/Users/qi/OneDrive - University of Leeds/projects/insar/turkey/"
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "decomposed")
    export_ve_vu = True
    os.makedirs(output_dir, exist_ok=True)
    
    
    #%% Define vn, los, los_sigma directories and file formats here.#
    
    #GNSS velocity interpolation files
    vn_dir = "gps_kriging/external_drift/"
    # vn_file = vn_dir+"vn_interpolated.tif"
    vn_file = os.path.join(vn_dir, 'vn_interpolated_upsampled.tif') #vn_dir+"surface_filter/vn_2_surface_filter_200_0.005d.tif"
    sn_file = os.path.join(vn_dir, 'sn_interpolated_upsampled.tif') #vn_dir+"sn_interpolated.tif"

    ### Define lists of tracks for ascending and descending tracks
    asc_0_list = ['014A_05232_242525', '014A_04920_161613', '014A_05540_151414', '043A_05221_121313', '043A_05008_161514', '043A_05421_141313']
    asc_1_list = ['087A_05101_131313', '087A_05317_121617', '116A_04978_131311', '116A_05207_252525', '116A_05367_141313', '116A_05582_141515', '145A_04956_121313', '145A_05152_131313', '145A_05351_131313']
    dsc_0_list = ['021D_04972_131213', '021D_05266_252525', '021D_05566_131313', '050D_05046_141313', '050D_05246_131313', '050D_05443_121313']
    dsc_1_list = ['123D_04925_130707', '123D_05095_141313', '123D_05292_242525', '123D_05489_131313', '094D_04913_101213', '094D_05100_131313', '094D_05288_130913']

    ## Define directories for LOS and sigma files
    los_dir = "1.velmap_interseismic/range_referenced/"
    sigma_dir = "1.velmap_interseismic/range_referenced/"
    # Define directories for SBOI files
    sboi_dir = "1.velmap_interseismic/sboi_referenced/"
    sboi_sigma_dir = "1.velmap_interseismic/sboi_referenced/"
    flip_sign = False
    
    # Define directory containing NEU coefs or incidence and heading angles in degrees
    args = {"data_dir": los_dir,
            "sig_dir": sigma_dir,
            "neu_dir": sigma_dir,
            "data_suffix": '_vel_GNSS_ref_frame.tif', 
            "sig_suffix": '.vstd_scaled.geo.tif',
            "n_suffix": '.N.geo.tif',
            "e_suffix": '.E.geo.tif',
            "u_suffix": '.U.geo.tif'}
    
    args_sboi = {"data_dir": sboi_dir,
            "sig_dir": sboi_sigma_dir,
            "neu_dir": sboi_sigma_dir,
            "data_suffix": '_vel_GNSS_ref_frame.tif', 
            "sig_suffix": '.vstd_scaled.geo.tif',
            "n_suffix": '.N.azi.geo.tif',
            "e_suffix": '.E.azi.geo.tif',
            "u_suffix": '.U.azi.geo.tif'}
        
    # set up groups
    los_asc_0 = Layer(asc_0_list, **args)
    los_asc_1 = Layer(asc_1_list, **args)
    los_dsc_0 = Layer(dsc_0_list, **args)
    los_dsc_1 = Layer(dsc_1_list, **args)
    sboi_asc_0 = Layer(asc_0_list, **args_sboi)
    sboi_asc_1 = Layer(asc_1_list, **args_sboi)
    sboi_dsc_0 = Layer(dsc_0_list, **args_sboi)
    sboi_dsc_1 = Layer(dsc_1_list, **args_sboi)

    layers_full = [los_asc_0, los_asc_1, los_dsc_0, los_dsc_1, sboi_asc_0, sboi_asc_1, sboi_dsc_0, sboi_dsc_1] #los_asc_0, los_asc_1, los_dsc_0, los_dsc_1,
    layers_los = [los_asc_0, los_asc_1, los_dsc_0, los_dsc_1]
    layers_sboi = [sboi_asc_0, sboi_asc_1, sboi_dsc_0, sboi_dsc_1]

    # define outer boundary of the data sets
    canvas = define_canvas_size(layers_full)
    for i, layer in enumerate(layers_full):
        layer.place_data_sig_N_E_U_to_canvas(canvas, plotting=True, layer_name=f"layer_{i+1}", output_dir=output_dir)
    
    
    #%% 3D solution
    print("3D decomposition starts...")
    
    # mask_3D = np.logical_and(
    #     must_have([los_asc_0, los_asc_1, los_dsc_0, los_dsc_1], at_least=2),
    #     must_have(layers_sboi, at_least=1)
    # )
    mask_3D = np.logical_and(
        np.logical_and(
            must_have([los_asc_0, los_asc_1], at_least=1),
            must_have([los_dsc_0, los_dsc_1], at_least=1)
        ),
        must_have(layers_sboi, at_least=1)
    )

    for layer in layers_full:
        layer.flatten_and_mask(mask_3D)

    # Inversion
    vn_mask, ve_mask, vu_mask, vn_sig_mask, ve_sig_mask, vu_sig_mask = decomposition_inversion(layers_full)
    vn = back_2d(vn_mask, mask_3D, canvas)
    ve = back_2d(ve_mask, mask_3D, canvas)
    vu = back_2d(vu_mask, mask_3D, canvas)
    vn_sig = back_2d(vn_sig_mask, mask_3D, canvas)
    ve_sig = back_2d(ve_sig_mask, mask_3D, canvas)
    vu_sig = back_2d(vu_sig_mask, mask_3D, canvas)

    #Export results to GeoTIFF
    export_tif(np.array(vn.data), canvas, os.path.join(output_dir, "north_3D.tif"))
    export_tif(np.array(ve.data), canvas, os.path.join(output_dir, "east_3D.tif"))
    export_tif(np.array(vu.data), canvas, os.path.join(output_dir, "up_3D.tif"))
    export_tif(np.array(vn_sig.data), canvas, os.path.join(output_dir, "north_sigma_3D.tif"))
    export_tif(np.array(ve_sig.data), canvas, os.path.join(output_dir, "east_sigma_3D.tif"))
    export_tif(np.array(vu_sig.data), canvas, os.path.join(output_dir, "up_sigma_3D.tif"))
            
    #%% 2D solution
    print("2D decomposition starts...")
    
    # Re-load clean, unmasked LOS data
    los_asc_0 = Layer(asc_0_list, **args)
    los_asc_1 = Layer(asc_1_list, **args)
    los_dsc_0 = Layer(dsc_0_list, **args)
    los_dsc_1 = Layer(dsc_1_list, **args)
    layers_los = [los_asc_0, los_asc_1, los_dsc_0, los_dsc_1]

    # Place them onto the full canvas again
    for i, layer in enumerate(layers_los):
        layer.place_data_sig_N_E_U_to_canvas(canvas, plotting=False, layer_name=f"2D_layer_{i}")
    
    # Remove Vn contribution from LOS and propagate sigma_n to sigma_los
    vn_gps = OpenTif(vn_file, sigfile=sn_file)
    for group in layers_los:
        remove_vn_contribution_from_los_and_propagate_sigma_n_to_sigma_los(group, vn_gps)
        
    mask_2D = np.logical_and(
        must_have([los_asc_0, los_asc_1], at_least=1),
        must_have([los_dsc_0, los_dsc_1], at_least=1)
    )
    mask_2D[mask_3D] = False  # ensure no overlap
    
    for layer in layers_los:
        layer.flatten_and_mask(mask_2D)

    # Run 2D inversion
    ve_mask_2d, vu_mask_2d, ve_sig_mask_2d, vu_sig_mask_2d = decomposition_inversion_eu(layers_los)
    ve_2d = back_2d(ve_mask_2d, mask_2D, canvas)
    vu_2d = back_2d(vu_mask_2d, mask_2D, canvas)
    ve_sig_2d = back_2d(ve_sig_mask_2d, mask_2D, canvas)
    vu_sig_2d = back_2d(vu_sig_mask_2d, mask_2D, canvas)
    
    #Export results to GeoTIFF
    export_tif(np.array(ve_2d), canvas, os.path.join(output_dir, "east_2D.tif"))
    export_tif(np.array(vu_2d), canvas, os.path.join(output_dir, "up_2D.tif"))
    export_tif(np.array(ve_sig_2d), canvas, os.path.join(output_dir, "east_sigma_2D.tif"))
    export_tif(np.array(vu_sig_2d), canvas, os.path.join(output_dir, "up_sigma_2D.tif"))
    

    # #%%plotting 3D solution
    # fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    # components = [ve, vn, vu]
    # titles = ['East Velocity (Ve)', 'North Velocity (Vn)', 'Up Velocity (Vu)']

    # for ax, data, title in zip(axes, components, titles):
    #     im = ax.imshow(data, interpolation='nearest', cmap='RdBu_r',
    #                 vmin=np.nanpercentile(data, 5),
    #                 vmax=np.nanpercentile(data, 95))
    #     ax.set_title(title, fontsize=14)
    #     ax.axis('off')
    #     cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #     cbar.set_label('mm/yr')

    # plt.tight_layout()
    # plt.savefig('' + output_dir + '/decomposed_3D_solution.png', dpi=300)
    # plt.close()
    # fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # sigma_components = [ve_sig, vn_sig, vu_sig]
    # titles = ['East Uncertainty (σVe)', 'North Uncertainty (σVn)', 'Up Uncertainty (σVu)']

    # for ax, data, title in zip(axes, sigma_components, titles):
    #     im = ax.imshow(data, interpolation='nearest', cmap='viridis',
    #                 vmin=0,
    #                 vmax=1)
    #     ax.set_title(title, fontsize=14)
    #     ax.axis('off')
    #     cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #     cbar.set_label('Uncertainty (mm/yr)')

    # plt.tight_layout()
    # plt.savefig('' + output_dir + '/decomposed_3D_solution_sigma.png', dpi=300)
    # # export_tif(vn, canvas, os.path.join(output_dir, "north{}.tif".format(suffix)))
    # # export_tif(ve, canvas, os.path.join(output_dir, "east{}.tif".format(suffix)))
    # # export_tif(vu, canvas, os.path.join(output_dir, "up{}.tif".format(suffix)))
    # # export_tif(vn_sig, canvas, os.path.join(output_dir, "north_sigma{}.tif".format(suffix)))
    # # export_tif(ve_sig, canvas, os.path.join(output_dir, "east_sigma{}.tif".format(suffix)))
    # # export_tif(vu_sig, canvas, os.path.join(output_dir, "up_sigma{}.tif".format(suffix)))
        
    # #%% Plotting 2D solution
    # fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # components = [ve_2d, vu_2d]
    # titles = ['East Velocity (Ve)', 'Up Velocity (Vu)']

    # for ax, data, title in zip(axes, components, titles):
    #     im = ax.imshow(data, interpolation='nearest', cmap='RdBu_r',
    #                 vmin=np.nanpercentile(data, 5),
    #                 vmax=np.nanpercentile(data, 95))
    #     ax.set_title(title, fontsize=14)
    #     ax.axis('off')
    #     cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #     cbar.set_label('mm/yr')

    # plt.tight_layout()
    # plt.savefig('' + output_dir + '/decomposed_2D_solution.png', dpi=300)
    # plt.close()

    # # Plotting 2D uncertainties
    # fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # components = [ve_sig_2d, vu_sig_2d]
    # titles = ['Ve_sigma', 'Vu_sigma']

    # for ax, data, title in zip(axes, components, titles):
    #     im = ax.imshow(data, interpolation='nearest', cmap='viridis',
    #                 vmin=np.nanpercentile(data, 5),
    #                 vmax=np.nanpercentile(data, 95))
    #     ax.set_title(title, fontsize=14)
    #     ax.axis('off')
    #     cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #     cbar.set_label('mm/yr')

    # plt.tight_layout()
    # plt.savefig('' + output_dir + '/decomposed_2D_solution_sigma.png', dpi=300)
    
    # #combine 2D and 3D results
    # ve_combined = np.where(~np.isnan(ve), ve, ve_2d)
    # vu_combined = np.where(~np.isnan(vu), vu, vu_2d)

    # # Plotting combined results
    # fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    # components = [ve_combined, vn, vu_combined]
    # titles = ['Ve Combined','Vn' ,'Vu Combined']

    # for ax, data, title in zip(axes, components, titles):
    #     im = ax.imshow(data, interpolation='nearest', cmap='RdBu_r',
    #                 vmin=np.nanpercentile(data, 5),
    #                 vmax=np.nanpercentile(data, 95))
    #     ax.set_title(title, fontsize=14)
    #     ax.axis('off')
    #     cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #     cbar.set_label('mm/yr')

    # plt.tight_layout()
    # plt.savefig('' + output_dir + '/decomposed_combined_solution.png', dpi=300)
