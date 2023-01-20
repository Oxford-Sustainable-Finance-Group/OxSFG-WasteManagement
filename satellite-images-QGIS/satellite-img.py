# Please note: this PyQGIS file will only in QGIS python console.
import pandas as pd
import numpy as np

def create_png(path, name):
    project = QgsProject.instance()
    manager = project.layoutManager()  
    layoutName = 'Layout1'
    layouts_list = manager.printLayouts()
    for layout in layouts_list:
        if layout.name() == layoutName:
            manager.removeLayout(layout)
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName(layoutName)
    manager.addLayout(layout)
    # create map item in the layout
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



path = "/Users/aloksingh/Documents/Oxford/Waste-management/plant-classification/model/results/clean/new-img/" # path to the folder where images will be saved
data = pd.read_csv("/Users/aloksingh/Documents/Oxford/Waste-management/plant-classification/model/results/clean/stage1-result/test_no_plant_result.txt", names= ['id','lat','long','p','p2'], delimiter='\t')

for i in range(len(data)):#len(data)
    id =  data.iloc[i][0]
    lat = data.iloc[i][1]
    long =data.iloc[i][2]
    name= str(int(id))+'_'+str(lat)+'_'+str(long)
    canvas = iface.mapCanvas()
    centre = canvas.extent().center()
    current_scale = canvas.scale()
    canvas.setCenter(QgsPointXY(np.float(long),np.float(lat)))
    canvas.zoomScale(500)# this will control the zoom level. It can be changed accordingly.  For zoomin the value should be closer to 0 and for zoom out the value should be high 
    create_png(path, name)