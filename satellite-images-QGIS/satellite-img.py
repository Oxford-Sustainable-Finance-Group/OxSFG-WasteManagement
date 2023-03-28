import pandas as pd
import numpy as np
from PyQt5.QtCore import QSize
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
    map.setRect(10, 10, 10, 10) 
    rect = iface.mapCanvas().extent()    
    rect.scale(0.1)
    iface.mapCanvas().resize(QSize(1157,149)) #used to fix the canvas so that it won't get changed with change in display size
    canvasSize = qgis.utils.iface.mapCanvas().size()    
    #print("Width : " + str(canvasSize.width()) + " / Height : " + str(canvasSize.height())) # to see the canvas size
    #iface.mapCanvas().zoomScale(500)
    map.setExtent(rect)
    
    #canvas.setExtent(rect)
    #canvas.refresh
    map.setBackgroundColor(QColor(255, 255, 255, 255))
    layout.addLayoutItem(map)
    map.attemptMove(QgsLayoutPoint(0, 0, QgsUnitTypes.LayoutMillimeters))
    map.attemptResize(QgsLayoutSize(297, 210, QgsUnitTypes.LayoutMillimeters))
        
    layout = manager.layoutByName(layoutName)
    exporter = QgsLayoutExporter(layout)

    fn = path + name
    exporter.exportToImage(fn + '.png', QgsLayoutExporter.ImageExportSettings())


path = "/Users/aloksingh/Documents/Oxford/Waste-management/USA/"

''' If you want to read Xlsx file use below command'''


data = pd.read_excel("/Users/aloksingh/Documents/Oxford/Waste-management/Lat_lon_USA.xlsx",usecols=["LAT_No_round","LON_No_round"])

''' If you want to read csv file use below command'''

#data = pd.read_csv("/Users/aloksingh/Documents/Oxford/Waste-management/Hydro-dataset-germany.csv",usecols=["LAT_WWTP","LON_WWTP"])

for i in range(len(data)):
    lat = data.iloc[i][0]
    long = data.iloc[i][1]
    name= str(i)+'_'+str(lat)+'_'+str(long)
    canvas = iface.mapCanvas()
    centre = canvas.extent().center()
    current_scale = canvas.scale()
    canvas.setCenter(QgsPointXY(np.float(long),np.float(lat)))
    #canvas.zoomScale(200)#same as rect.scale(0.1)
    #canvas.setRotation(180)
    create_png(path, name)
