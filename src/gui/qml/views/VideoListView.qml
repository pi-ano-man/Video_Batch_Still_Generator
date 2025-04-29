import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import Qt.labs.platform as Platform  // Alternativer Ansatz für StandardPaths

Rectangle {
    id: root
    color: "white"  // Weißer Hintergrund für den gesamten Tab
    
    property var videos: []
    property bool processing: false
    
    signal processVideo(string path)
    
    // Hilfsfunktion: Konvertiert URLs oder Strings zu lokalen Dateipfaden
    function addFiles(pathsOrUrls) {
        if (!pathsOrUrls || pathsOrUrls.length === 0) {
            return;
        }
        
        // Fortschrittsanzeige einblenden
        showProgressOverlay();
        
        var localPaths = [];
        for (var i = 0; i < pathsOrUrls.length; i++) {
            var item = pathsOrUrls[i];
            // Prüfen, ob wir mit einem URL-Objekt oder einem String arbeiten
            if (typeof item === 'object' && item.toString().indexOf('file://') === 0) {
                // URL-Objekt mit toLocalFile-Methode
                if (typeof item.toLocalFile === 'function') {
                    localPaths.push(item.toLocalFile());
                } else {
                    // URL-String ohne Methode
                    var urlStr = item.toString();
                    // Einfache Konvertierung von URL zu Pfad (für macOS/Linux)
                    localPaths.push(urlStr.replace('file://', ''));
                }
            } else {
                // Schon ein String-Pfad
                localPaths.push(item);
            }
        }
        
        if (localPaths.length > 0) {
            // Verwende ein Timer, um die UI nicht zu blockieren
            var timer = Qt.createQmlObject('import QtQuick; Timer {}', root);
            timer.interval = 100;
            timer.triggered.connect(function() {
                mainController.add_files(localPaths);
                // Fortschrittsanzeige wieder ausblenden
                hideProgressOverlay();
                timer.destroy();
            });
            timer.start();
        } else {
            hideProgressOverlay();
        }
    }
    
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 20
        
        // Header mit Aktionsbuttons, nur sichtbar wenn Videos vorhanden sind
        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            visible: videoList.count > 0  // Nur anzeigen, wenn Videos vorhanden sind
            
            Item { Layout.fillWidth: true }  // Flexibler Platzhalter für Rechtsausrichtung
            
            Button {
                text: qsTr("Alle verarbeiten")
                enabled: !root.processing
                visible: videoList.count > 0 && mainController.get_people().length > 0
                
                onClicked: {
                    if (mainController.get_people().length > 0) {
                        var selectedPerson = mainController.get_current_person();
                        if (selectedPerson && selectedPerson.length > 0) {
                            mainController.process_directory(selectedPerson);
                        } else {
                            // Zeige Fehler oder wähle erste Person aus
                            var people = mainController.get_people();
                            if (people.length > 0) {
                                mainController.set_current_person(people[0]);
                                mainController.process_directory(people[0]);
                            }
                        }
                    }
                }
            }
            
            Button {
                text: qsTr("Alle entfernen")
                visible: videoList.count > 0
                onClicked: {
                    confirmDeleteAllDialog.open()
                }
            }
        }
        
        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredHeight: videoList.count > 0 ? implicitHeight : 0
            clip: true
            
            ListView {
                id: videoList
                model: videos
                delegate: videoDelegate
                spacing: 2
            }
        }
        
        // DropArea für Drag & Drop und Klickfläche
        Rectangle {
            id: dropZone
            Layout.fillWidth: true
            Layout.fillHeight: videoList.count === 0  // Expandiert bei leerer Liste
            Layout.minimumHeight: 120                 // Mindesthöhe
            color: dragArea.containsDrag ? "#e3f2fd" : "#f5f5f5"
            border.color: dragArea.containsDrag ? "#1976d2" : "#bdbdbd"
            border.width: 2
            radius: 8
            
            RowLayout {
                anchors.centerIn: parent
                spacing: 20
                
                Text {
                    text: qsTr("Videos oder Ordner hierher ziehen oder auswählen:")
                    color: "#1976d2"
                    font.pixelSize: 16
                }
                
                Button {
                    text: qsTr("Dateien auswählen")
                    onClicked: fileDialog.open()
                }
                Button {
                    text: qsTr("Ordner auswählen")
                    onClicked: folderDialog.open()
                }
            }
            
            DropArea {
                id: dragArea
                anchors.fill: parent
                keys: ["text/uri-list"]
                onEntered: function(drag) { 
                    drag.acceptProposedAction(); 
                }
                onDropped: function(drag) {
                    if (drag && drag.urls) {
                        root.addFiles(drag.urls);
                    }
                }
            }
        }
    }
    
    // Einfacher FileDialog ohne fileMode-Property
    FileDialog {
        id: fileDialog
        title: qsTr("Videos auswählen")
        currentFolder: Platform.StandardPaths.writableLocation(Platform.StandardPaths.MoviesLocation)
        fileMode: FileDialog.OpenFiles  // Mehrfachauswahl ermöglichen
        nameFilters: [
            qsTr("Videos (*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv *.mpeg *.mpg)")
        ]
        onAccepted: {
            root.addFiles(fileDialog.selectedFiles);
        }
    }
    
    FolderDialog {
        id: folderDialog
        title: qsTr("Ordner auswählen")
        onAccepted: {
            if (folderDialog.selectedFolder) {
                root.addFiles([folderDialog.selectedFolder]);
            }
        }
    }
    
    // Bestätigungsdialog für "Alle entfernen"
    Dialog {
        id: confirmDeleteAllDialog
        title: qsTr("Alle Videos entfernen")
        standardButtons: Dialog.Yes | Dialog.No
        modal: true
        
        // Zentrale Positionierung des Dialogs
        anchors.centerIn: Overlay.overlay
        width: Math.min(parent.width * 0.8, 400)  // Maximal 80% Breite oder 400px
        height: contentColumn.implicitHeight + 100  // Automatische Höhe plus Platz
        
        // Hintergrund für deutlichere Abgrenzung
        background: Rectangle {
            color: "white"
            border.color: "#cccccc"
            border.width: 1
            radius: 5
        }
        
        // Spalten-Layout für bessere Strukturierung
        ColumnLayout {
            id: contentColumn
            anchors.fill: parent
            anchors.margins: 20
            spacing: 20
            
            Text {
                Layout.fillWidth: true
                text: qsTr("Möchten Sie wirklich alle Videos entfernen?")
                color: "#555555"
                font.pixelSize: 14
                wrapMode: Text.WordWrap
                horizontalAlignment: Text.AlignHCenter
            }
        }
        
        onAccepted: {
            mainController.remove_all_videos()
        }
    }
    
    Component {
        id: videoDelegate
        
        Rectangle {
            width: videoList.width
            height: 90  // Erhöht für 16:9 Thumbnails
            color: index % 2 === 0 ? "#f9f9f9" : "#ffffff"
            border.color: "#eeeeee"
            border.width: 1
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 10
                
                // Thumbnail-Bild im 16:9 Format
                Rectangle {
                    Layout.preferredWidth: 160
                    Layout.preferredHeight: 90  // 16:9 Verhältnis
                    Layout.alignment: Qt.AlignCenter
                    color: "#f0f0f0"
                    
                    Image {
                        id: thumbnailImage
                        anchors.fill: parent
                        anchors.margins: 2
                        source: ""
                        fillMode: Image.PreserveAspectFit
                        asynchronous: true
                        
                        // Platzhalter während Thumbnail generiert wird
                        Rectangle {
                            anchors.fill: parent
                            color: "#e0e0e0"
                            visible: thumbnailImage.status !== Image.Ready
                            
                            Text {
                                anchors.centerIn: parent
                                text: "🎬"
                                font.pixelSize: 24
                                color: "#888888"
                            }
                        }
                        
                        Component.onCompleted: {
                            // Thumbnail asynchron laden
                            var thumbnailPath = mainController.get_video_thumbnail(modelData);
                            if (thumbnailPath) {
                                // Der Pfad ist bereits absolut, braucht nur file:// vorne
                                thumbnailImage.source = "file://" + thumbnailPath;
                                console.log("Thumbnail-Pfad: " + thumbnailImage.source);
                            }
                        }
                    }
                }
                
                // Video-Informationen
                Column {
                    Layout.fillWidth: true
                    spacing: 4
                    
                    Text {
                        text: modelData.split('/').pop()  // Extrahiere Dateinamen
                        font.pixelSize: 14
                        font.bold: true
                        elide: Text.ElideMiddle
                        width: parent.width
                    }
                    
                    Text {
                        text: modelData
                        font.pixelSize: 10
                        color: "#888888"
                        elide: Text.ElideMiddle
                        width: parent.width
                    }
                }
                
                // Aktionen
                RowLayout {
                    spacing: 5
                    
                    // Verarbeiten-Button
                    Button {
                        text: qsTr("Verarbeiten")
                        enabled: !root.processing
                        
                        onClicked: {
                            root.processVideo(modelData)
                        }
                    }
                    
                    // Entfernen-Button (X)
                    Button {
                        text: "✕"
                        font.pixelSize: 14
                        Layout.preferredWidth: 32
                        Layout.preferredHeight: 32
                        
                        ToolTip.text: qsTr("Video entfernen")
                        ToolTip.visible: hovered
                        ToolTip.delay: 500
                        
                        onClicked: {
                            mainController.remove_video(modelData)
                        }
                    }
                }
            }
        }
    }
    
    // Funktion zum Anzeigen der Fortschrittsanzeige
    function showProgressOverlay() {
        progressOverlay.visible = true;
    }
    
    // Funktion zum Ausblenden der Fortschrittsanzeige
    function hideProgressOverlay() {
        progressOverlay.visible = false;
    }
    
    // Fortschrittsanzeige für Drag & Drop und Dateioperationen
    // Muss am Ende des Dokuments stehen und mit hohem z-Wert
    Rectangle {
        id: progressOverlay
        anchors.fill: parent
        color: "#80000000"  // Halbdurchsichtiges Schwarz
        visible: false
        z: 1000  // Sehr hoher z-Wert, um über allen anderen Elementen zu sein
        
        BusyIndicator {
            anchors.centerIn: parent
            running: parent.visible
            width: 48
            height: 48
        }
        
        Text {
            anchors.centerIn: parent
            anchors.verticalCenterOffset: 40
            color: "white"
            font.pixelSize: 14
            text: qsTr("Dateien werden hinzugefügt...")
        }
    }
} 