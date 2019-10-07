import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def get_corr_maps3x3(fA, fB, queries):

    queries = queries.numpy()
    N_batch = queries.shape[0]
    Nsamples = queries.shape[1]
    correlations = torch.zeros(N_batch, Nsamples, 64, 64)


    for n in range(N_batch):

        fBI = fB[n]
        d1 = fBI * fBI
        e_norms1 = torch.sum(d1, dim=0)
        e_norms1 = e_norms1.repeat([32, 1, 1])
        fBI_norm = fBI/ e_norms1
        fBI_norm = fBI_norm.view(1, 32, 64, 64)

        fAI = fA[n]
        d2 = fAI * fAI
        e_norms2 = torch.sum(d2, dim=0)
        e_norms2 = e_norms2.repeat([32, 1, 1])
        fAI_norm = fAI / e_norms2

        for s in range(Nsamples):

            filter = torch.zeros(32, 3, 3).cuda()
            y, x = int(queries[n, s, 0]), int(queries[n, s, 1])


            miny, maxy = max(0, y - 1), min(y + 1, 63)
            minx, maxx = max(0, x - 1), min(x + 1, 63)
            sfy, efy = 0, 2
            sfx, efx = 0, 2

            if y == 0:
                sfy = 1
                efy = 2
            if y == 63:
                sfy = 0
                efy = 1
            if x == 0:
                sfx = 1
                efx = 2
            if x == 63:
                sfx = 0
                efx = 1

            filter[:, sfy:efy+1, sfx:efx+1] = fAI_norm[:, miny:maxy+1, minx:maxx+1]
            corr = F.conv2d(fBI_norm,  filter.view(1, 32, 3, 3), padding=1)
            #filter[:, sfy:efy+1, sfx:efx+1] =  fA[n, :, miny:maxy+1, minx:maxx+1]
            #corr = F.conv2d(fB[n].view(1, 32, 64, 64),  fA[n, :, y, x].view(1, 32, 1, 1))
            #corr /= 64*64
            corr = corr.view(64, 64).reshape(64*64)
            corr = torch.softmax(corr, dim=0)
            correlations[n, s] = corr.reshape(64, 64)

    return correlations.cuda()

def get_corr_maps5x5(fA, fB, queries):

    queries = queries.numpy()
    N_batch = queries.shape[0]
    Nsamples = queries.shape[1]
    correlations = torch.zeros(N_batch, Nsamples, 64, 64)


    for n in range(N_batch):

        fBI = fB[n]
        d1 = fBI * fBI
        e_norms1 = torch.sum(d1, dim=0)
        e_norms1 = e_norms1.repeat([32, 1, 1])
        fBI_norm = fBI/ e_norms1
        fBI_norm = fBI_norm.view(1, 32, 64, 64)

        fAI = fA[n]
        d2 = fAI * fAI
        e_norms2 = torch.sum(d2, dim=0)
        e_norms2 = e_norms2.repeat([32, 1, 1])
        fAI_norm = fAI / e_norms2

        for s in range(Nsamples):

            filter = torch.zeros(32, 5, 5).cuda()
            y, x = int(queries[n, s, 0]), int(queries[n, s, 1])
            for ir, r in enumerate(range(-2, 3)):
                if y + r < 0 or y + r > 63:
                    continue
                for ic, c in enumerate(range(-2, 3)):
                    if x + c < 0 or x + c > 63:
                        continue
                    filter[:, ir, ic] = fAI_norm[:, y+r, x+c]

            corr = F.conv2d(fBI_norm,  filter.view(1, 32, 5, 5), padding=2)
            corr = corr.view(64, 64).reshape(64*64)
            corr = F.softmax(corr, dim=0)
            correlations[n, s] = corr.reshape(64, 64)

    return correlations.cuda()

def get_corr_maps3x3_fast(fA, fB, queries, scores):

    queries = queries.numpy()
    N_batch = queries.shape[0]
    Nsamples = queries.shape[1]
    correlations = torch.zeros(N_batch, Nsamples, 64, 64).cuda()

    for n in range(N_batch):

        fBI = fB[n][0]
        d1 = fBI * fBI
        e_norms1 = torch.sum(d1, dim=0)
        e_norms1 = e_norms1.repeat([32, 1, 1])
        fBI_norm = fBI/ e_norms1
        fBI_norm = fBI_norm.view(1, 32, 64, 64)

        fAI = fA[n][0]
        d2 = fAI * fAI
        e_norms2 = torch.sum(d2, dim=0)
        e_norms2 = e_norms2.repeat([32, 1, 1])
        fAI_norm = fAI / e_norms2

        filters = torch.zeros(Nsamples, 32, 3, 3).cuda()

        for s in range(Nsamples):

            y, x = int(queries[n, s, 0]), int(queries[n, s, 1])

            if y < 0 or y > 63 or x < 0 or x > 63 or (scores[s]/2) < 0.1:
                continue


            miny, maxy = max(0, y - 1), min(y + 1, 63)
            minx, maxx = max(0, x - 1), min(x + 1, 63)
            sfy, efy = 0, 2
            sfx, efx = 0, 2

            if y == 0:
                sfy = 1
                efy = 2
            if y == 63:
                sfy = 0
                efy = 1
            if x == 0:
                sfx = 1
                efx = 2
            if x == 63:
                sfx = 0
                efx = 1

            filters[s][:, sfy:efy+1, sfx:efx+1] = fAI_norm[:, miny:maxy+1, minx:maxx+1]

        with torch.no_grad():
            corr = F.conv2d(fBI_norm,  filters, padding=1)
            corr = corr.view(17, 64, 64).reshape(17, 64*64)
            corr = torch.softmax(corr, dim=1)
            correlations[n] = corr.reshape(17, 64, 64)

    return correlations

