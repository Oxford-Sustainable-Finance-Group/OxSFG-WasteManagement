import math
import pandas as pd
import numpy as np

R = 6378.1 #Radius of the Earth
d = 0.14 #Distance in km

d_set = [ 45, 90, 135,180,225,270,315,360]
#d_set = [45, 90, 180, 270, 360]

def create_png( path, name):
    project = QgsProject.instance()
    root = project.layerTreeRoot()
    manager = project.layoutManager()
    
    layoutName = 'Layout1'
    layouts_list = manager.printLayouts()
    # remove any duplicate layouts
    for layout in layouts_list:
        if layout.name() == layoutName:
            manager.removeLayout(layout)
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName(layoutName)
    manager.addLayout(layout)
    map = QgsLayoutItemMap(layout)
    map.setRect(20, 20, 20, 20) 
    rect = iface.mapCanvas().extent()
    map.setExtent(rect)
    map.setBackgroundColor(QColor(255, 255, 255, 0))
    layout.addLayoutItem(map)
    map.attemptMove(QgsLayoutPoint(0, 0, QgsUnitTypes.LayoutMillimeters))
    map.attemptResize(QgsLayoutSize(297, 210, QgsUnitTypes.LayoutMillimeters))
        
    layout = manager.layoutByName(layoutName)
    exporter = QgsLayoutExporter(layout)

    fn = path + name
    exporter.exportToImage(fn + '.png', QgsLayoutExporter.ImageExportSettings())

path = "/Users/aloksingh/Documents/Oxford/Waste-management/plant-classification/model/results/clean/plants-to-correct/"

data = pd.read_csv("/Users/aloksingh/Documents/Oxford/Waste-management/plant-classification/model/results/clean/Romania/new-img_plant_result copy.txt",names=['id','lat','lon','p'],delimiter='\t')

for j in range(len(data)):#len(data)
    id = data.iloc[j][0]
    lat = data.iloc[j][1]
    long = data.iloc[j][2]
    name= str(int(id))+'_'+str(lat)+'_'+str(long)
    canvas = iface.mapCanvas()
    centre = canvas.extent().center()
    current_scale = canvas.scale()
    canvas.setCenter(QgsPointXY(np.float(long),np.float(lat)))
    canvas.zoomScale(800)
    create_png(path, name)
    lat1 = math.radians(lat)
    lon1 = math.radians(long)
    for i, de in enumerate(d_set):
        brng = math.radians(de)
        lat2 = math.asin( math.sin(lat1)*math.cos(d/R) +
        math.cos(lat1)*math.sin(d/R)*math.cos(brng))#the lat result I'm hoping for
        lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat1),
             math.cos(d/R)-math.sin(lat1)*math.sin(lat2))
        lat = math.degrees(lat2)
        long = math.degrees(lon2)
        name= str(int(id))+'_'+str(lat)+'_'+str(long)
        canvas = iface.mapCanvas()
        centre = canvas.extent().center()
        current_scale = canvas.scale()
        canvas.setCenter(QgsPointXY(np.float(long),np.float(lat)))
        canvas.zoomScale(800)
        create_png(path, name)
