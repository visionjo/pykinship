import numpy as np
from html4vision import Col, imagetable
from src.configs import CONFIGS

import os
import numpy as np
import bobo.app
import bobo.util
from bobo.util import istuple, islist, isnumpy, quietprint
from bobo.image import Image
from bobo.video import VideoDetection
from janus.template import GalleryTemplate
from collections import defaultdict
import PIL

try:
    from pyspark.rdd import PipelinedRDD
except ImportError:
    pass


def montage(
    imset,
    m,
    n,
    crop=False,
    skip=True,
    grayscale=True,
    do_plot=False,
    figure=None,
    border=0,
    border_bgr=(128, 128, 128),
    do_flush=False,
):
    """Square grayscale montage image of images of size (m,n)"""
    n_imgs = len(imset)
    M = int(np.ceil(np.sqrt(n_imgs)))
    N = M
    padding = (M + 1) * border
    size = (M * m + padding, N * n + padding)
    bc = border_bgr
    if grayscale:
        if islist(bc) or istuple(bc) or isnumpy(bc):
            bc = np.mean(bc)
        I = np.array(PIL.Image.new(mode="L", size=size, color=bc))
    else:
        I = np.array(PIL.Image.new(mode="RGB", size=size, color=bc))
    k = 0
    for i in range(M):
        for j in range(N):
            if k >= n_imgs:
                break
            sliceM, sliceN = i * (m + border) + border, j * (n + border) + border
            try:
                if crop:
                    if not imset[k].bbox.valid():
                        print(
                            '[janus.visualize.montage] invalid bounding box "%s" '
                            % str(imset[k].bbox)
                        )
                        if skip == False:
                            print("[janus.visualize.montage] using original image")
                            if grayscale:
                                im = imset[k].grayscale().resize(n, m).data
                            else:
                                im = imset[k].resize(n, m).data
                        else:
                            raise
                    else:
                        if grayscale:
                            im = (
                                imset[k]
                                .grayscale()
                                .crop(imset[k].bbox)
                                .resize(n, m)
                                .data
                            )
                        else:
                            im = imset[k].crop(imset[k].bbox).resize(n, m).data
                else:
                    if grayscale:
                        im = imset[k].grayscale().resize(n, m).data  # m=width, n=height
                    else:
                        im = imset[k].resize(n, m).data

                if im.dtype == np.float32:
                    if im.max() <= 1.0:
                        im *= 255.0
                    im = im.astype(np.uint8)

                I[sliceN : sliceN + n, sliceM : sliceM + m] = im

            except KeyboardInterrupt:
                raise
            except:
                print("[janus.visualize.montage] skipping...")
                if skip:
                    pass
                else:
                    raise

            if do_flush:
                imset[k].flush()  # clear memory
            k += 1

    if k == 0:
        print("[janus.visualize.montage] Warning: No images were processed")

    if do_plot is True:
        im = Image("")
        im.data = I
        # HACK: float(0-255) graycale images display incorrectly
        if grayscale:
            im.preprocess().rgb().show(figure=figure)
        else:
            im.show(figure=figure)

    return I


def probegalleryhtml(
    y_flat,
    yhat_flat,
    probe_ids,
    gallery_ids,
    probeset,
    galleryset,
    testname,
    results_dir=bobo.app.results(),
    do_samediff=False,
    max_results=20,
    do_bbox=True,
    sightings=None,
):
    # FIXME: Handle RDDs, list(GalleryTemplate), list(Image*)
    if (
        "PipelinedRDD" in globals()
        and isinstance(galleryset, PipelinedRDD)
        and isinstance(galleryset.take(1)[0], GalleryTemplate)
    ):
        galleryset = galleryset.keyBy(
            lambda tmpl: tmpl.media()[0].attributes["TEMPLATE_ID"]
        ).collect()
        probeset = probeset.keyBy(
            lambda tmpl: tmpl.media()[0].attributes["TEMPLATE_ID"]
        ).collect()
        galleryset = {k: v for (k, v) in galleryset}
        probeset = {k: v for (k, v) in probeset}
    else:
        dd = defaultdict(list)
        for im in galleryset:
            dd[im.attributes["TEMPLATE_ID"]].append(im)
        galleryset = {k: GalleryTemplate(media=imgs) for (k, imgs) in dd.iteritems()}

        dd = defaultdict(list)
        for im in probeset:
            dd[im.attributes["TEMPLATE_ID"]].append(im)
        probeset = {k: GalleryTemplate(media=imgs) for (k, imgs) in dd.iteritems()}

    num_gal = len(gallery_ids)
    num_probe = len(probe_ids)
    num_tests = num_gal * num_probe
    assert len(y_flat) == num_tests, "num_tests=%d, num_y=%d" % (num_tests, len(y_flat))
    assert len(yhat_flat) == num_tests, "num_tests=%d, num_yhat=%d" % (
        num_tests,
        len(yhat_flat),
    )

    y = np.array(y_flat).reshape((num_probe, num_gal))
    yhat = np.array(yhat_flat).reshape((num_probe, num_gal))

    def nlind(lvl, str):
        return "\n" + "\t" * lvl + str

    outpathname = os.path.join(results_dir, testname)
    bobo.util.remkdir(outpathname)
    quietprint("writing to %s" % outpathname)
    detailpath = os.path.join(outpathname, "detail")
    bobo.util.remkdir(detailpath)

    indexfile = os.path.join(outpathname, "index.html")
    summary = open(indexfile, "w", 0)
    summary.write(nlind(0, "<html>"))
    summary.write(nlind(0, "<table border=1>"))

    summary.write(nlind(0, "<tr>"))
    summary.write(nlind(0, "<th>link</th>"))
    summary.write(nlind(0, "<th>score</th>"))
    summary.write(nlind(0, "<th>probe template id</th>"))
    summary.write(nlind(0, "<th>probe subject id</th>"))
    if do_samediff:
        summary.write(nlind(0, "<th>same/diff</th>"))
    summary.write(nlind(0, "<th>probe & gallery</th>"))
    summary.write(nlind(0, "</tr>"))

    def writeImg(detailpath, im, do_bbox, sightings):
        sighting_id = im.attributes["SIGHTING_ID"]
        outimg = os.path.join(detailpath, "%s.png" % sighting_id)
        if os.path.exists(outimg):
            return
        if do_bbox:
            color = (
                (0, 196, 0)
                if sightings is not None
                and sighting_id in sightings
                and sightings[sighting_id] > 0.0
                else (196, 0, 0)
            )
            im.clone().rgb().rescale(scale=150.0 / im.height()).drawbox(
                color=color
            ).saveas(outimg)
        else:
            im.clone().rgb().resize(rows=150).saveas(outimg)

    # Write out probe and gallery imagery
    for tmplset in (probeset, galleryset):
        num_items = len(tmplset)
        for (k, (tid, tmpl)) in enumerate(tmplset.iteritems()):
            if num_items % 50 == 0:
                quietprint("[janus.visualize]: Processed %d/%d" % (k, num_items), 1)

            for x in tmpl:
                if isinstance(x, VideoDetection):
                    for im in x.frames():
                        writeImg(detailpath, im, do_bbox, sightings)
                else:
                    im = x
                    try:
                        writeImg(detailpath, im, do_bbox, sightings)
                    except:
                        print(im)
                        raise

    # Iterate over probe and gallery sets, write out html by score
    for (ridx, pid) in enumerate(probe_ids):
        probe_template = probeset[pid]
        probe_subject_id = probe_template.category()
        pfilename = os.path.join("detail", "%s.html" % pid)

        scores = yhat[ridx, :]
        assert len(scores) == num_gal
        mask = y[ridx, :]
        assert len(mask) == num_gal

        (sorted_scores, sorted_mask, sorted_gids) = (
            list(x)
            for x in zip(
                *sorted(
                    zip(scores, mask, gallery_ids), key=(lambda p: p[0]), reverse=True
                )
            )
        )

        best_gallery_template = galleryset[sorted_gids[0]]
        best_gallery_same = sorted_mask[0]

        summary.write(nlind(0, "<tr>"))
        summary.write(nlind(0, '<td><a href="%s">link</a></td>' % pfilename))
        summary.write(nlind(0, "<td>%1.03f</td>" % sorted_scores[0]))
        summary.write(nlind(0, "<td>%s</td>" % pid))
        summary.write(nlind(0, "<td>%s</td>" % probe_subject_id))
        if do_samediff:
            if int(best_gallery_same) > 0:
                summary.write(
                    nlind(0, '<td style="background-color: #227722">same</td>')
                )
            else:
                summary.write(
                    nlind(0, '<td style="background-color: #992222">diff</td>')
                )
        summary.write(nlind(0, "<td>probe<hr/>"))
        for x in probe_template:
            if isinstance(x, VideoDetection):
                for im in x:
                    sighting_id = im.attributes["SIGHTING_ID"]
                    summary.write(
                        nlind(
                            0,
                            '<img src="%s" />'
                            % os.path.join("detail", "%s.png" % sighting_id),
                        )
                    )
            else:
                im = x
                sighting_id = im.attributes["SIGHTING_ID"]
                summary.write(
                    nlind(
                        0,
                        '<img src="%s" />'
                        % os.path.join("detail", "%s.png" % sighting_id),
                    )
                )
        summary.write(nlind(0, "<br/><hr/>gallery<hr/>"))
        for x in best_gallery_template:
            if isinstance(x, VideoDetection):
                for im in x:
                    sighting_id = im.attributes["SIGHTING_ID"]
                    summary.write(
                        nlind(
                            0,
                            '<img src="%s" />'
                            % os.path.join("detail", "%s.png" % sighting_id),
                        )
                    )
            else:
                im = x
                sighting_id = im.attributes["SIGHTING_ID"]
                summary.write(
                    nlind(
                        0,
                        '<img src="%s" />'
                        % os.path.join("detail", "%s.png" % sighting_id),
                    )
                )

        summary.write(nlind(0, "</td>"))
        summary.write(nlind(0, "</tr>"))

        # now do detailed pages
        with open(os.path.join(outpathname, pfilename), "w", 0) as detail:
            detail.write(nlind(0, "<html>"))
            detail.write(nlind(0, "<table border=1>"))
            detail.write(nlind(0, "<tr>"))
            detail.write(nlind(0, "<th>score</th>"))
            detail.write(nlind(0, "<th>gallery template id</th>"))
            detail.write(nlind(0, "<th>gallery subject id</th>"))
            if do_samediff:
                detail.write(nlind(0, "<th>same/diff</th>"))
            detail.write(nlind(0, "<th>probe & gallery</th>"))
            detail.write(nlind(0, "</tr>"))
            detail.write(nlind(0, "<tr>"))
            colspan = 4 if do_samediff is True else 3
            detail.write(
                nlind(
                    0,
                    '<td colspan="%d">Probe: %s, %s</td>'
                    % (colspan, pid, probe_subject_id),
                )
            )
            detail.write(nlind(0, "<td>"))
            for x in probe_template:
                if isinstance(x, VideoDetection):
                    for im in x:
                        sighting_id = im.attributes["SIGHTING_ID"]
                        detail.write(
                            nlind(0, '<img src="%s" />' % "%s.png" % sighting_id)
                        )
                else:
                    im = x
                    sighting_id = im.attributes["SIGHTING_ID"]
                    detail.write(nlind(0, '<img src="%s" />' % "%s.png" % sighting_id))
            detail.write(nlind(0, "</td>"))
            detail.write(nlind(0, "</tr>"))
            for (k, (score, same, gid)) in enumerate(
                zip(sorted_scores, sorted_mask, sorted_gids)
            ):  # already sorted by score
                if k >= max_results:  # Limit result set size
                    break
                gallery_template = galleryset[gid]
                gallery_subject_id = gallery_template.category()
                # assert((gallery_subject_id == probe_subject_id) == (int(same) > 0))
                detail.write(nlind(0, "<tr>"))
                detail.write(nlind(0, "<td>%1.03f</td>" % score))
                detail.write(nlind(0, "<td>%s</td>" % gid))
                detail.write(nlind(0, "<td>%s</td>" % gallery_subject_id))
                if do_samediff:
                    if int(same) > 0:
                        detail.write(
                            nlind(0, '<td style="background-color: #227722">same</td>')
                        )
                    else:
                        detail.write(
                            nlind(0, '<td style="background-color: #992222">diff</td>')
                        )
                detail.write(nlind(0, "<td>"))
                for x in gallery_template:
                    if isinstance(x, VideoDetection):
                        for im in x:
                            sighting_id = im.attributes["SIGHTING_ID"]
                            detail.write(
                                nlind(0, '<img src="%s" />' % "%s.png" % sighting_id)
                            )
                    else:
                        im = x
                        sighting_id = im.attributes["SIGHTING_ID"]
                        detail.write(
                            nlind(0, '<img src="%s" />' % "%s.png" % sighting_id)
                        )
                detail.write(nlind(0, "</td>"))
                detail.write(nlind(0, "</tr>"))

            detail.write(nlind(0, "</table>\n"))
            detail.write(nlind(0, "</html>\n"))

    summary.write(nlind(0, "</table>\n"))
    summary.write(nlind(0, "</html>\n"))
    summary.close()

    url = "file://%s" % indexfile
    quietprint("[janus.visualize] Completed web page generation: %s" % url, 2)

    return url


def create_html_page_face_montage(p1, p2, labels, scores, dir_fold):
    neg_ids = np.where(labels == 0)[0][-100:]
    pos_ids = np.where(labels == 1)[0][:100]

    p1_pos = p1[pos_ids]
    p1_neg = p1[neg_ids]
    p2_pos = p2[pos_ids]
    p2_neg = p2[neg_ids]

    hard_pos_pairs = np.array(list(zip(p1_pos, p2_pos)))
    hard_neg_pairs = np.array(list(zip(p1_neg, p2_neg)))

    cols = [
        Col("text", "FID1", p1_pos),
        Col(
            "img",
            "P1",
            [CONFIGS.path.dfid + f + ".jpg" for f in list(hard_pos_pairs[:, 0])],
        ),
        Col(
            "img",
            "P2",
            [CONFIGS.path.dfid + f + ".jpg" for f in list(hard_pos_pairs[:, 1])],
        ),
        Col("text", "Scores", ["{0:0.5}".format(sc * 100) for sc in scores[pos_ids]]),
    ]
    # cols2 = []
    imagetable(
        cols,
        imscale=0.75,  # scale all images to 50%
        sticky_header=True,  # keep the header on the top
        out_file=dir_fold + "hard_positives.html",
        style="img {border: 1px solid black;-webkit-box-shadow: 2px 2px 1px #ccc; box-shadow: 2px 2px 1px #ccc;}",
    )
    cols = [
        Col(
            "text",
            "FID1",
            ["{}\n{}".format(tt1, tt2) for (tt1, tt2) in zip(p1_neg, p2_neg)],
            ["{}\n{}".format(tt1, tt2) for (tt1, tt2) in zip(p1_neg, p2_neg)],
        ),
        Col(
            "img",
            "P1",
            [CONFIGS.path.dfid + f + ".jpg" for f in list(hard_neg_pairs[:, 0])],
        ),
        Col(
            "img",
            "P2",
            [CONFIGS.path.dfid + f + ".jpg" for f in list(hard_neg_pairs[:, 1])],
        ),
        # Col('text', 'FID2', t2[neg_ids]),
        Col("text", "Scores", ["{0:0.5}".format(sc * 100) for sc in scores[neg_ids]]),
    ]

    imagetable(
        cols,
        imscale=0.75,  # scale all images to 50%
        sticky_header=True,  # keep the header on the top
        out_file=dir_fold + "hard_negatives.html",
        style="img {border: 1px solid black;-webkit-box-shadow: 2px 2px 1px #ccc; box-shadow: 2px 2px 1px #ccc;}",
    )
