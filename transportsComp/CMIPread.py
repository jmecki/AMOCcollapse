###############################################################################
#
# Reads in data from CMIP files, making adjustments:
#
###############################################################################

import cftime
import calendar

import numpy as np
from netCDF4 import Dataset

# Determine starting file and position:
def timeFile(model,cal,units2,tt,Files):
    ff    = 0
    nn    = np.nan
    found = False
    dd = cftime.num2date(tt,units2,cal)
    
    nf = len(Files)
    while(((ff < nf) & (found == False))):
        ncid  = Dataset(Files[ff],'r')
        ttf   = ncid.variables['time'][:]
        units = ncid.variables['time'].units
        ncid.close()
        if ((model == 'FGOALS-g2')):
            units = (units + '-01')
        
        dd2 = cftime.date2num(dd,units,cal)
        
        if(((dd2 >= ttf[0]) & (dd2 <= ttf[-1]))):
            nn = 0
            while(((nn < len(ttf)) & (found == False))):
                if ((cftime.num2date(ttf[nn],units,cal).year  == cftime.num2date(tt,units2,cal).year) &
                    (cftime.num2date(ttf[nn],units,cal).month == cftime.num2date(tt,units2,cal).month)):
                    found = True
                else:
                    nn = nn + 1
        else:
            ff = ff + 1          
    return ff, nn

# Checks and fixes time if calendars don't match:
def fixTime(units,units2,cal,time,time_bnds):
    if units2 != units:
        yr_init = int(units.split(' ')[2].split('-')[0])
        yr_new  = int(units2.split(' ')[2].split('-')[0])
        nleap   = 0 
        if ((cal == 'standard') | (cal == 'gregorian')):
            days = 365
            for yy in range(yr_init,yr_new):
                if calendar.isleap(yy):
                    nleap = nleap + 1
        elif cal == 'noleap':
            days = 365
        else:
            days = int(cal[0:3])
        offset = days*(yr_new-yr_init) + nleap
        time      = time      + offset
        time_bnds = time_bnds + offset
        
    return time, time_bnds


###  Ocean Data ###
###############################################################################

# Reads in lat and lon data from ocean:
def Olatlon(cdict,model,infile,var):

    ncid = Dataset(infile,'r')
    # Determine name of lat and lon:
    if ((model == 'MPI-ESM1-2-XR') & ((var == 'uo') | (var == 'vo'))):
        if (var == 'uo'):
            lon = ncid.variables['lon_2'][:,:]
            lat = ncid.variables['lat_2'][:,:]
        elif (var == 'vo'):
            lon = ncid.variables['lon_3'][:,:]
            lat = ncid.variables['lat_3'][:,:]
    else:
        vark = ncid.variables.keys()
        for vak in vark:
            if ((vak == 'lat') | (vak == 'latitude') | (vak == 'nav_lat')):
                nlat = vak
            if ((vak == 'lon') | (vak == 'longitude') | (vak == 'nav_lon')):
                nlon = vak

        if cdict['reg']:
            lon = ncid.variables[nlon][:]
            lat = ncid.variables[nlat][:]
        else:
            lon = ncid.variables[nlon][:,:]
            lat = ncid.variables[nlat][:,:]
    ncid.close()
    
    # Flip North-South:
    if cdict['flipNS']:
        lat  = np.flip(lat,axis=0)
        if not cdict['reg']:
            lon  = np.flip(lon,axis=0)
            
        
    # Extra row in T fields (coded only for regular grid):
    if cdict['extraT']:
        if ((var == 'vo') | (var == 'uo') | (var == 'tauuo') | (var == 'tauvo') | (var == 'hfy')):
            if cdict['reg']:
                lat = np.concatenate((lat,[-90,]),0)
            else:
                lat = np.concatenate((lat,-90*np.ones((1,np.size(lat,axis=1)),'float')),0)
                lon = np.concatenate((lon,lon[-1:,:]),0)
                
                
    # Remove extra W-E columns:
    if cdict['reg']:
        ni  = np.size(lon,axis=0)
        lon = lon[cdict['haloEW'][0]:(ni-cdict['haloEW'][1])]
    else:
        ni  = np.size(lon,axis=1)
        lon = lon[:,cdict['haloEW'][0]:(ni-cdict['haloEW'][1])]
        lat = lat[:,cdict['haloEW'][0]:(ni-cdict['haloEW'][1])]
        
    return lon,lat

# Reads in data from a 2D ocean field:
def Oread2Ddata(cdict,model,infile,var,time=None,lev=None,mask=False):
    ncid = Dataset(infile,'r')
    # Flip Up-Down:
    if ((lev != None) & ((model == 'CFSv2-2011') | (model == 'FGOALS-gl') | (model == 'HadGEM2-AO'))):
        nk  = len(ncid.variables['lev'][:])
        lev = nk - 1 - lev
    if mask:
        if time == None:
            if lev == None:
                data = 1-np.squeeze(ncid.variables[var][:,:]).mask
            else:
                data = 1-np.squeeze(ncid.variables[var][lev,:,:]).mask
        else:
            if lev == None:
                data = 1-np.squeeze(ncid.variables[var][time,:,:]).mask
            else:
                data = 1-np.squeeze(ncid.variables[var][time,lev,:,:]).mask
    else:
        if time == None:
            if lev == None:
                data = np.squeeze(ncid.variables[var][:,:]).data
            else:
                data = np.squeeze(ncid.variables[var][lev,:,:]).data
        else:
            if lev == None:
                data = np.squeeze(ncid.variables[var][time,:,:]).data
            else:
                data = np.squeeze(ncid.variables[var][time,lev,:,:]).data
    ncid.close()
    
    # Flip North-South:
    if cdict['flipNS']:
        data = np.flip(data,axis=0)
        
    # Extra row in u and v fields:
    if cdict['extraT']:
        if ((var == 'vo') | (var == 'uo') | (var == 'tauvo') | (var == 'tauuo') | (var == 'hfy')):
            data = np.concatenate((data,np.expand_dims(data[-1,:],0)),0)
                
    # Remove extra W-E columns:
    ni   = np.size(data,axis=1)
    data = data[:,cdict['haloEW'][0]:(ni-cdict['haloEW'][1])]
        
    return data

# Reads in data from a 3D ocean field:
def Oread3Ddata(cdict,model,infile,var,time=None,mask=False):
    ncid = Dataset(infile,'r')
    if mask:
        if time == None:
            data = 1-np.squeeze(ncid.variables[var][:,:,:]).mask
        else:
            data = 1-np.squeeze(ncid.variables[var][time,:,:,:]).mask
    else:
        if time == None:
            data = np.squeeze(ncid.variables[var][:,:,:]).data
        else:
            data = np.squeeze(ncid.variables[var][time,:,:,:]).data
    ncid.close()
    
    # Flip North-South:
    if cdict['flipNS']:
        data = np.flip(data,axis=1)
        
    # Extra row in u and v fields:
    if cdict['extraT']:
        if ((var == 'vo') | (var == 'uo') | (var == 'tauvo') | (var == 'tauuo') | (var == 'hfy')):
            data = np.concatenate((data,np.expand_dims(data[:,-1,:],1)),1)
            
    # Remove extra W-E columns:
    ni   = np.size(data,axis=2)
    data = data[:,:,cdict['haloEW'][0]:(ni-cdict['haloEW'][1])]
    
    # Flip Up-Down:
    if ((model == 'CFSv2-2011') | (model == 'FGOALS-gl') | (model == 'HadGEM2-AO')):
        data = data[::-1,:,:]
        
    return data

###  Atmosphere Data ###
###############################################################################

###  Moving Data ###
###############################################################################

# Move data onto different grid points:
def moveData(cdict,meshmask,grid1,grid2,data,computation='mean',dyt=[]): 
    if ((grid1 == 'T') & (grid2 == 'U')):
        if cdict['grid'][0] == 'B':
            if np.size(np.shape(data)) == 2:
                tmp = np.tile(data,(4,1,1))
                ncid  = Dataset(meshmask,'r')
                umask = ncid.variables['umask'][0,:,:]
                ncid.close()
                if cdict['grid'][1] == 'b':
                    tmp[2,:-1,:] = data[1:,:]
                    tmp[3,:-1,:] = data[1:,:]
                elif cdict['grid'][1] == 't':
                    tmp[2,1:,:]  = data[:-1,:]
                    tmp[3,1:,:]  = data[:-1,:]
                if cdict['grid'][2] == 'l':
                    tmp[1,:,:] = np.roll(tmp[0,:,:],1,axis=1)
                    tmp[3,:,:] = np.roll(tmp[2,:,:],1,axis=1)
                elif cdict['grid'][2] == 'r':
                    tmp[1,:,:] = np.roll(tmp[0,:,:],-1,axis=1)
                    tmp[3,:,:] = np.roll(tmp[2,:,:],-1,axis=1)                
            elif np.size(np.shape(data)) == 3:
                tmp = np.tile(data,(4,1,1,1))
                ncid  = Dataset(meshmask,'r')
                umask = ncid.variables['umask'][:,:,:]
                ncid.close()
                if cdict['grid'][1] == 'b':
                    tmp[2,:,:-1,:] = data[:,1:,:]
                    tmp[3,:,:-1,:] = data[:,1:,:]
                elif cdict['grid'][1] == 't':
                    tmp[2,:,1:,:]  = data[:,:-1,:]
                    tmp[3,:,1:,:]  = data[:,:-1,:]
                if cdict['grid'][2] == 'l':
                    tmp[1,:,:,:] = np.roll(tmp[0,:,:,:],1,axis=2)
                    tmp[3,:,:,:] = np.roll(tmp[2,:,:,:],1,axis=2)
                elif cdict['grid'][2] == 'r':
                    tmp[1,:,:,:] = np.roll(tmp[0,:,:,:],-1,axis=2)
                    tmp[3,:,:,:] = np.roll(tmp[2,:,:,:],-1,axis=2)
        elif cdict['grid'][0] == 'C':
            if np.size(np.shape(data)) == 2:
                tmp = np.tile(data,(2,1,1))
                ncid  = Dataset(meshmask,'r')
                umask = ncid.variables['umask'][0,:,:]
                ncid.close()
                if cdict['grid'][2] == 'l':
                    tmp[1,:,:] = np.roll(tmp[0,:,:],1,axis=1)
                elif cdict['grid'][2] == 'r':
                    tmp[1,:,:] = np.roll(tmp[0,:,:],-1,axis=1)                
            elif np.size(np.shape(data)) == 3:
                tmp = np.tile(data,(2,1,1,1))
                ncid  = Dataset(eshmask,'r')
                umask = ncid.variables['umask'][:,:,:]
                ncid.close()
                if cdict['grid'][2] == 'l':
                    tmp[1,:,:,:] = np.roll(tmp[0,:,:,:],1,axis=2)
                elif cdict['grid'][2] == 'r':
                    tmp[1,:,:,:] = np.roll(tmp[0,:,:,:],-1,axis=2)
        elif cdict['grid'][0] == 'A':
            if np.size(np.shape(data)) == 2:
                tmp = np.tile(data,(1,1,1)) 
                ncid  = Dataset(meshmask,'r')
                umask = ncid.variables['umask'][0,:,:]
                ncid.close()              
            elif np.size(np.shape(data)) == 3:
                tmp = np.tile(data,(1,1,1,1))
                ncid  = Dataset(meshmask,'r')
                umask = ncid.variables['umask'][:,:,:]
                ncid.close()
                
        if computation == 'mean':
            datanew = np.squeeze(np.mean(tmp,axis=0)*umask)
        elif computation == 'min':
            datanew = np.squeeze(np.min(tmp,axis=0)*umask)
        elif computation == 'max':
            datanew = np.squeeze(np.max(tmp,axis=0)*umask)
            
    elif ((grid1 == 'T') & (grid2 == 'VT')):
        # Data won't be masked:
        ncid  = Dataset(meshmask,'r')
        dyt   = ncid.variables['dyt'][:,:]
        ncid.close()
        if ((cdict['grid'][0] == 'A') | (cdict['grid'][1] == 'b')):
            if np.size(np.shape(data)) == 2:
                tmp = np.tile(data,(2,1,1))
                dyt = np.tile(dyt,(2,1,1))
                tmp[1,1:,:] = tmp[0,:-1,:]
                dyt[1,1:,:] = dyt[0,:-1,:]
            elif np.size(np.shape(data)) == 3:
                tmp = np.tile(data,(2,1,1,1))
                dyt = np.tile(dyt,(2,np.size(tmp,1),1,1))
                tmp[1,:,1:,:] = tmp[0,:,:-1,:]
                dyt[1,:,1:,:] = dyt[0,:,:-1,:] 
        else:
            if np.size(np.shape(data)) == 2:
                tmp = np.tile(data,(2,1,1))
                dyt = np.tile(dyt,(2,1,1))
                tmp[1,:-1,:] = tmp[0,1:,:]
                dyt[1,:-1,:] = dyt[0,1:,:]
            elif np.size(np.shape(data)) == 3:
                tmp = np.tile(data,(2,1,1,1))
                dyt = np.tile(dyt,(2,np.size(tmp,1),1,1))
                tmp[1,:,:-1,:] = tmp[0,:,1:,:]
                dyt[1,:,:-1,:] = dyt[0,:,1:,:] 
                
            
        if computation == 'mean':
            datanew = np.squeeze(np.sum(tmp*dyt,axis=0)/np.sum(dyt,axis=0))
        elif computation == 'min':
            datanew = np.squeeze(np.min(tmp,axis=0))
        elif computation == 'max':
            datanew = np.squeeze(np.max(tmp,axis=0))
        
    elif ((grid1 == 'V') & (grid2 == 'VT')):
        # Data won't be masked:
        if cdict['grid'][0] == 'A':
            ncid  = Dataset(meshmask,'r')
            dyv   = ncid.variables['dyv'][:,:]
            ncid.close()
            if np.size(np.shape(data)) == 2:
                tmp = np.tile(data,(2,1,1))
                dyv = np.tile(dyv,(2,1,1))
                tmp[1,1:,:] = tmp[0,:-1,:]
                dyv[1,1:,:] = dyv[0,:-1,:]
            elif np.size(np.shape(data)) == 3:
                tmp = np.tile(data,(2,1,1,1))
                dyv = np.tile(dyv,(2,np.size(tmp,1),1,1))
                tmp[1,:,1:,:] = tmp[0,:,:-1,:]
                dyv[1,:,1:,:] = dyv[0,:,:-1,:] 
            
            if computation == 'mean':
                datanew = np.squeeze(np.sum(tmp*dyv,axis=0)/np.sum(dyv,axis=0))
            elif computation == 'min':
                datanew = np.squeeze(np.min(tmp,axis=0))
            elif computation == 'max':
                datanew = np.squeeze(np.max(tmp,axis=0))
                
        elif cdict['grid'][0] == 'B':
            if cdict['grid'][2] == 'r':
                if np.size(np.shape(data)) == 2:
                    tmp = np.tile(data,(2,1,1))
                    tmp[1,:,:] = np.roll(tmp[0,:,:],1,axis=1)
                elif np.size(np.shape(data)) == 3:
                    tmp = np.tile(data,(2,1,1,1))
                    tmp[1,:,:,:] = np.roll(tmp[0,:,:,:],1,axis=2)
            if cdict['grid'][2] == 'l':
                if np.size(np.shape(data)) == 2:
                    tmp = np.tile(data,(2,1,1))
                    tmp[1,:,:] = np.roll(tmp[0,:,:],-1,axis=1)
                elif np.size(np.shape(data)) == 3:
                    tmp = np.tile(data,(2,1,1,1))
                    tmp[1,:,:,:] = np.roll(tmp[0,:,:,:],-1,axis=2)
            
            if computation == 'mean':
                datanew = np.squeeze(np.mean(tmp,axis=0))
            elif computation == 'min':
                datanew = np.squeeze(np.min(tmp,axis=0))
            elif computation == 'max':
                datanew = np.squeeze(np.max(tmp,axis=0))
        elif Model.Ogrid[0] == 'C':
            datanew = tmp
            
    elif ((grid1 == 'T') & (grid2 == 'V')):
        # Data won't be masked:
        if (cdict['grid'][0] == 'A'):
            datanew = data
        elif (cdict['grid'][0] == 'B'):
            if np.size(np.shape(data)) == 2:
                tmp = np.tile(data,(4,1,1))
            elif np.size(np.shape(data)) == 3:
                tmp = np.tile(data,(4,1,1,1))
                
            if (cdict['grid'][1] == 't'):
                if np.size(np.shape(data)) == 2:
                    tmp[2,:-1,:] = tmp[0,1:,:]
                    tmp[3,:-1,:] = tmp[0,1:,:]
                elif np.size(np.shape(data)) == 3:
                    tmp[2,:,:-1,:] = tmp[0,:,1:,:]
                    tmp[3,:,:-1,:] = tmp[0,:,1:,:]
            elif (cdict['grid'][1] == 'b'):
                if np.size(np.shape(data)) == 2:
                    tmp[2,1:,:] = tmp[0,:-1,:]
                    tmp[3,1:,:] = tmp[0,:-1,:]
                elif np.size(np.shape(data)) == 3:
                    tmp[2,:,1:,:] = tmp[0,:,:-1,:]
                    tmp[3,:,1:,:] = tmp[0,:,:-1,:]
            
            if (cdict['grid'][2] == 'r'):
                if np.size(np.shape(data)) == 2:
                    tmp[1,:,:] = np.roll(tmp[0,:,:],-1,axis=1)
                    tmp[2,:,:] = np.roll(tmp[3,:,:],-1,axis=1)
                elif np.size(np.shape(data)) == 3:
                    tmp[1,:,:,:] = np.roll(tmp[0,:,:,:],-1,axis=2)
                    tmp[2,:,:,:] = np.roll(tmp[3,:,:,:],-1,axis=2)
            elif (cdict['grid'][2] == 'l'):
                if np.size(np.shape(data)) == 2:
                    tmp[1,:,:] = np.roll(tmp[0,:,:],1,axis=1)
                    tmp[2,:,:] = np.roll(tmp[3,:,:],1,axis=1)
                elif np.size(np.shape(data)) == 3:
                    tmp[1,:,:,:] = np.roll(tmp[0,:,:,:],1,axis=2)
                    tmp[2,:,:,:] = np.roll(tmp[3,:,:,:],1,axis=2)
            
            if computation == 'mean':
                datanew = np.squeeze(np.mean(tmp,axis=0))
            elif computation == 'min':
                datanew = np.squeeze(np.min(tmp,axis=0))
            elif computation == 'max':
                datanew = np.squeeze(np.max(tmp,axis=0))
        else:
            if (np.size(dyt) == 0):
                ncid  = Dataset(meshmask,'r')
                dyt   = ncid.variables['dyt'][:,:]
                ncid.close()
            if ((cdict['grid'][1] == 'b')):
                if np.size(np.shape(data)) == 2:
                    tmp = np.tile(data,(2,1,1))
                    dyt = np.tile(dyt,(2,1,1))
                    tmp[1,1:,:] = tmp[0,:-1,:]
                    dyt[1,1:,:] = dyt[0,:-1,:]
                elif np.size(np.shape(data)) == 3:
                    tmp = np.tile(data,(2,1,1,1))
                    dyt = np.tile(dyt,(2,np.size(tmp,1),1,1))
                    tmp[1,:,1:,:] = tmp[0,:,:-1,:]
                    dyt[1,:,1:,:] = dyt[0,:,:-1,:] 
            else:
                if np.size(np.shape(data)) == 2:
                    tmp = np.tile(data,(2,1,1))
                    dyt = np.tile(dyt,(2,1,1))
                    tmp[1,:-1,:] = tmp[0,1:,:]
                    dyt[1,:-1,:] = dyt[0,1:,:]
                elif np.size(np.shape(data)) == 3:
                    tmp = np.tile(data,(2,1,1,1))
                    dyt = np.tile(dyt,(2,np.size(tmp,1),1,1))
                    tmp[1,:,:-1,:] = tmp[0,:,1:,:]
                    dyt[1,:,:-1,:] = dyt[0,:,1:,:] 
                
            
            if computation == 'mean':
                datanew = np.squeeze(np.sum(tmp*dyt,axis=0)/np.sum(dyt,axis=0))
            elif computation == 'min':
                datanew = np.squeeze(np.min(tmp,axis=0))
            elif computation == 'max':
                datanew = np.squeeze(np.max(tmp,axis=0))
            
    else:
        print('Need to code')
    
    return datanew