// ----------------------------------------------------------
// SENTINEL CAMPAIGNS

//  GLOBALS & CONFIGURATION LIST 
var CONFIG = {
  halfWindow:     20,
  aoiCloudThresh: 0.10,
  srScale:        60,
  exportScale:    10
};

var campaigns = [
  {"name": "BAS_2025",  "bbox": [4.6235, 44.2034, 4.8088, 44.7715], "date": "2025-07-22"},
  {"name": "BRC_2022",  "bbox": [5.5370, 45.6079, 5.6693, 45.7161], "date": "2022-07-20"},
  {"name": "BRC_2023",  "bbox": [5.5330, 45.6079, 5.6677, 45.7161], "date": "2023-07-17"},
  {"name": "DZM_2019",  "bbox": [4.6432, 44.2953, 4.6988, 44.4460], "date": "2019-06-26"},
  {"name": "DZM_2023",  "bbox": [4.6399, 44.2110, 4.7113, 44.5493], "date": "2023-07-12"},
  {"name": "HAUT_2025", "bbox": [5.4079, 45.5908, 5.8598, 45.9916], "date": "2025-07-01"},
  {"name": "PDR_2023",  "bbox": [4.7350, 45.2837, 4.8831, 45.7157], "date": "2023-07-18"}
];

// 2. SHARED PREPROCESSING FUNCTIONS
var maskCloudsAndShadows = function(img) {
  var scl = img.select('SCL');
  return img.updateMask(
    scl.neq(3)   // cloud shadow
       .and(scl.neq(8))  // cloud medium probability
       .and(scl.neq(9))  // cloud high probability
       .and(scl.neq(10)) // thin cirrus
  );
};

var addAOICloudFraction = function(aoi) {
  return function(img) {
    var scl = img.select('SCL');
    var isCloudOrShadow = scl.eq(3).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10));
    var frac = isCloudOrShadow.reduceRegion({
      reducer:   ee.Reducer.mean(),
      geometry:  aoi,
      scale:     CONFIG.srScale,
      maxPixels: 1e8
    }).get('SCL');
    return img.set('AOI_CLOUD_FRACTION', frac);
  };
};

// 3. LOOP OVER ALL CAMPAIGNS FOR EXPORT 
campaigns.forEach(function(site, index) {
  
  var aoi        = ee.Geometry.Rectangle(site.bbox);
  var centerDate = ee.Date(site.date);
  var startDate  = centerDate.advance(-CONFIG.halfWindow, 'day');
  var endDate    = centerDate.advance( CONFIG.halfWindow, 'day');

  // Build Collection
  var s2Col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate(startDate, endDate)
    .map(maskCloudsAndShadows)
    .map(addAOICloudFraction(aoi))
    .filter(ee.Filter.lt('AOI_CLOUD_FRACTION', CONFIG.aoiCloudThresh))
    .map(function(img) { return img.clip(aoi); });

  // Essential log prints
  print(site.name + ' — Scenes Found:', s2Col.size());
  print(site.name + ' — Dates Used:', s2Col.aggregate_array('system:time_start')
    .map(function(t) { return ee.Date(t).format('YYYY-MM-dd'); }));

  // Median Composite
  var median = s2Col.median();

  // SR Scaling -> [0, 1]
  var s2Cube = median
    .select(
      ['B2',    'B3',    'B4',  'B8',  'B11'],
      ['Blue', 'Green', 'Red', 'NIR', 'SWIR1']
    )
    .multiply(0.0001)
    .toFloat();  

  // Spectral Indices
  var ndvi = s2Cube.normalizedDifference(['NIR',   'Red'  ]).rename('NDVI');
  var ndwi = s2Cube.normalizedDifference(['Green', 'NIR'  ]).rename('NDWI');
  var ndmi = s2Cube.normalizedDifference(['NIR',   'SWIR1']).rename('NDMI');

  // High-Res IGN 1m DEM
  var dem = ee.Image('IGN/RGE_ALTI/1M/2_0/FXX')
    .select(['MNT'], ['DEM'])
    .clip(aoi)
    .toFloat();

  // 9-Band Stack forced into EPSG:2154 
  var finalStack = s2Cube
    .addBands([ndvi, ndwi, ndmi, dem])
    .reproject({ crs: 'EPSG:2154', scale: CONFIG.exportScale });

  // AUTOMATED EXPORT TASK
  Export.image.toDrive({
    image: finalStack,
    description: site.name + '_aux', // Outputs exactly: PDR_2023_aux
    folder: 'SISR_aux_data',
    scale: CONFIG.exportScale,
    region: aoi,
    crs: 'EPSG:2154',
    maxPixels: 1e13
  });

  // Visually sample the last index item
  if (index === campaigns.length - 1) {
    Map.centerObject(aoi, 11);
    Map.addLayer(finalStack, {bands: ['Red', 'Green', 'Blue'], min: 0.02, max: 0.18}, site.name + ' Sample RGB');
  }
});

// ---------------------------------------------------------------------
// LANDSAT CAMPAIGNS
// 1. GLOBALS & CONFIGURATION (NATIVE 30M GRID)
var CONFIG = {
  halfWindow:     20,
  aoiCloudThresh: 0.15, 
  srScale:        30,   
  exportScale:    30    
};

var campaigns = [
  {"name": "DZM_2013",  "bbox": [4.6350, 44.1432, 4.7346, 44.4709], "date": "2013-07-25"},
  {"name": "DZM_2014",  "bbox": [4.6378, 44.2068, 4.7148, 44.4467], "date": "2014-06-22"},
  {"name": "PDR_2013",  "bbox": [4.7325, 45.2729, 4.8221, 45.4149], "date": "2013-07-16"},
  {"name": "PDR_2014",  "bbox": [4.7337, 45.2711, 4.8193, 45.4144], "date": "2014-07-16"}
];

// 2. CLOUD & SHADOW MASKING FUNCTION 
var maskLandsatClouds = function(img) {
  var qa = img.select('QA_PIXEL');
  var dilatedCloud = 1 << 1;
  var cirrus       = 1 << 2;
  var cloud        = 1 << 3;
  var shadow       = 1 << 4;
  
  var mask = qa.bitwiseAnd(dilatedCloud).eq(0)
           .and(qa.bitwiseAnd(cirrus).eq(0))
           .and(qa.bitwiseAnd(cloud).eq(0))
           .and(qa.bitwiseAnd(shadow).eq(0));
           
  return img.updateMask(mask);
};

var addAOILandsatCloudFraction = function(aoi) {
  return function(img) {
    var qa = img.select('QA_PIXEL');
    var dilatedCloud = 1 << 1;
    var cirrus       = 1 << 2;
    var cloud        = 1 << 3;
    var shadow       = 1 << 4;
    var isCloudOrShadow = qa.bitwiseAnd(dilatedCloud).neq(0)
                        .or(qa.bitwiseAnd(cirrus).neq(0))
                        .or(qa.bitwiseAnd(cloud).neq(0))
                        .or(qa.bitwiseAnd(shadow).neq(0));
                        
    var frac = isCloudOrShadow.reduceRegion({
      reducer:   ee.Reducer.mean(),
      geometry:  aoi,
      scale:     CONFIG.srScale,
      maxPixels: 1e8
    }).get('QA_PIXEL');
    return img.set('AOI_CLOUD_FRACTION', frac);
  };
};

// 3. LOOP OVER ALL CAMPAIGNS FOR EXPORT 
campaigns.forEach(function(site, index) {
  
  var aoi        = ee.Geometry.Rectangle(site.bbox);
  var centerDate = ee.Date(site.date);
  var startDate  = centerDate.advance(-CONFIG.halfWindow, 'day');
  var endDate    = centerDate.advance( CONFIG.halfWindow, 'day');

  // Build Collection
  var l8Col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    .filterBounds(aoi)
    .filterDate(startDate, endDate)
    .map(maskLandsatClouds)
    .map(addAOILandsatCloudFraction(aoi))
    .filter(ee.Filter.lt('AOI_CLOUD_FRACTION', CONFIG.aoiCloudThresh))
    .map(function(img) { return img.clip(aoi); });

  // Native, essential pipeline log printouts
  print(site.name + ' — Scenes Found:', l8Col.size());

  // Median Composite
  var median = l8Col.median();

  // Corrected Landsat 8 Band Selection (B2 to B6)
  var scaledSpectral = median.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6'])
    .multiply(0.0000275).add(-0.2);

  // Structural Alignment renaming to match Sentinel-2 precisely
  var l8Cube = scaledSpectral.select(
    ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6'],
    ['Blue', 'Green', 'Red', 'NIR', 'SWIR1']
  ).toFloat();

  // Spectral Indices
  var ndvi = l8Cube.normalizedDifference(['NIR',   'Red'  ]).rename('NDVI');
  var ndwi = l8Cube.normalizedDifference(['Green', 'NIR'  ]).rename('NDWI');
  var ndmi = l8Cube.normalizedDifference(['NIR',   'SWIR1']).rename('NDMI');

  // High-Res IGN 1m DEM — Downsample smoothly to 30m using mean aggregation
  var dem = ee.Image('IGN/RGE_ALTI/1M/2_0/FXX')
    .select(['MNT'], ['DEM'])
    .clip(aoi)
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 1024
    })
    .toFloat();

  // 9-Band Stack forced into EPSG:2154 projection grid at 30m
  var finalStack = l8Cube
    .addBands([ndvi, ndwi, ndmi, dem])
    .reproject({ crs: 'EPSG:2154', scale: CONFIG.exportScale });

  // ── AUTOMATED EXPORT TASK (TARGETING SENTINEL WORKSPACE FOLDER) ──
  Export.image.toDrive({
    image: finalStack,
    description: site.name + '_aux', // Outputs exactly: PDR_2014_aux
    folder: 'S2_Auxiliary_data',  // Identical destination folder
    scale: CONFIG.exportScale,
    region: aoi,
    crs: 'EPSG:2154',
    maxPixels: 1e13
  });

  // Visually sample the last loop item map layout for validation
  if (index === campaigns.length - 1) {
    Map.centerObject(aoi, 11);
    Map.addLayer(finalStack, {bands: ['Red', 'Green', 'Blue'], min: 0.02, max: 0.18}, site.name + ' Aligned RGB');
  }
});