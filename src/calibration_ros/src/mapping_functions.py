def first_return_segmentation(gray, threshold,centre_x,centre_y):

    cutoff_radius=10
    radius_ivus=centre_x-cutoff_radius
    crop_ivus_index=80

    closest_pixel=[]
        
    #pull out all the voxels in each angular direction
    for theta in np.arange(0,2*np.pi,2*np.pi/200):
        
        try:
            #2D voxel traversal
            strtPt=np.asarray([centre_x,centre_y])
            endPt_x=centre_x+((radius_ivus)*np.cos(theta))
            endPt_y=centre_y+((radius_ivus)*np.sin(theta))
            endPt=[endPt_x,endPt_y]
            #not yielding an ordered list
            this_gridline=getIntersectPts(strtPt, endPt, geom=[0,1,0,0,0,1])
            this_gridline=np.asarray(list(this_gridline))
            this_gridline=np.round(this_gridline.astype(int))
            
            delta=this_gridline-[centre_x,centre_y]
            distances=np.linalg.norm(delta,axis=1)
            sort_indices=np.argsort(distances)
                    
            ordered_gridline=this_gridline[sort_indices]

            # gridlines.append(ordered_gridline)
            # print("ordered gridline", ordered_gridline)

            ordered_intensities=gray[ordered_gridline[:,0],ordered_gridline[:,1]]
            cropped_intensities=ordered_intensities[crop_ivus_index:]
            thresholded=cropped_intensities>threshold
            hyp_indices=np.squeeze(np.argwhere(thresholded))
            
            if(len(hyp_indices)!=0):
                min_d=np.min(hyp_indices)+crop_ivus_index #maybe more efficient to pull out first element
                closest_pixel.append([ordered_gridline[min_d]])
        except:
            pass

    return closest_pixel

def getIntersectPts(strPt, endPt, geom=[0,1,0,0,0,1]):

    x0 = geom[0]
    y0 = geom[3]

    (sX, sY) = (strPt[0], strPt[1])
    (eX, eY) = (endPt[0], endPt[1])
    xSpace = geom[1]
    ySpace = geom[5]

    sXIndex = ((sX - x0) / xSpace)
    sYIndex = ((sY - y0) / ySpace)
    eXIndex = ((eX - sXIndex) / xSpace) + sXIndex
    eYIndex = ((eY - sYIndex) / ySpace) + sYIndex


    dx = (eXIndex - sXIndex)
    dy = (eYIndex - sYIndex)
    xHeading = 1.0 if dx > 0 else -1.0 if dx < 0 else 0.0
    yHeading = 1.0 if dy > 0 else -1.0 if dy < 0 else 0.0

    xOffset = (1 - (math.modf(sXIndex)[0]))
    yOffset = (1 - (math.modf(sYIndex)[0]))

    ptsIndexes = []
    x = sXIndex
    y = sYIndex
    pt = (x, y) #1st pt

    if dx != 0:
        m = (float(dy) / float(dx))
        b = float(sY - sX * m )

    dx = abs(int(dx))
    dy = abs(int(dy))

    if dx == 0:
        for h in range(0, dy + 1):
            pt = (x, y + (yHeading *h))
            ptsIndexes.append(pt)

        return ptsIndexes


    #snap to half a cell size so we can find intersections on cell boundaries
    sXIdxSp = round(2.0 * sXIndex) / 2.0
    sYIdxSp = round(2.0 * sYIndex) / 2.0
    eXIdxSp = round(2.0 * eXIndex) / 2.0
    eYIdxSp = round(2.0 * eYIndex) / 2.0
    # ptsIndexes.append(pt)
    prevPt = False
    #advance half grid size
    for w in range(0, dx * 4):
        x = xHeading * (w / 2.0) + sXIdxSp
        y = (x * m + b)
        if xHeading < 0:
            if x < eXIdxSp:
                break
        else:
            if x > eXIdxSp:
                break

        pt = (round(x), round(y)) #snapToGrid
        # print(w, x, y)

        if prevPt != pt:
            ptsIndexes.append(pt)
            prevPt = pt
    #advance half grid size
    for h in range(0, dy * 4):
        y = yHeading * (h / 2.0) + sYIdxSp
        x = ((y - b) / m)
        if yHeading < 0:
            if y < eYIdxSp:
                break
        else:
            if y > eYIdxSp:
                break
        pt = (round(x), round(y)) # snapToGrid
        # print(h, x, y)

        if prevPt != pt:
            ptsIndexes.append(pt)
            prevPt = pt

    return set(ptsIndexes) #elminate duplicates