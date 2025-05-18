Anleitung:

python files wie gewohnt unter catkin_ws/python_files ablegen

Starten der Controller in Gazebo:
- roscore
- roslaunch om_position_controller roslaunch open_manipulator_6dof_gazebo open_manipulator_6dof_gazebo.launch controller:=position
- roslaunch open_manipulator_6dof_controller open_manipulator_position_gazebo_controller.launch



Zuerst muss mittels dmp_learn.py eine Trajektorie eines Rosbags gelernt werden.

Info (nicht unbedingt wichtig):
Der Rosbag "pick_2.bag" eignet sich einigermaßen gut als Pick-Trajektorie. (pick.bag habe ich zuerst versucht, das hat nicht so gute Ergebnisse geliefert.)
Der Rosbag "pick_2.bag" wurde aus move_cube3_forward.bag mittels: 
"rosbag filter recordings/move_cube3_forward.bag recordings/pick_2.bag "t.to_sec() < 1745491260.110637" erzeugt.
Der Zeitpunkt zum filtern (-> 1745491260.110637) wurde ermittelt, indem der rosbag "move_cube3_forward.bag" mittels "rosbag play recordings/move_cube3_forward.bag --hz=50" abgespielt wurde und dann mit der Leertaste pausiert wurde, sobald der Roboter in etwa den Würfel aufnimmt.

Die Syntax zum verwenden von learn_dmp.py lautet "python3 python_files/learn_dmp.py <path_to_rosbag> <name_of_dmp>" (name_of_dmp ist der gewünschte Name ohne dateiendung).
Die dmp für pick_2.bag habe ich mit python3 python_files/learn_dmp.py recordings/pick_2.bag "pick_2" erzeugt.

Ich würde für das ausprobieren auch empfehlen, eine dmp für das zurückfahren in die startposition zu erzeugen:
python3 python_files/learn_dmp.py recordings/part3.bag "return" (part3.bag wurde ähnlich wie pick_2.bag erzeugt)


Um die dmp nun zu verwenden kann folgender Befehl verwendet werden:
python3 python_files/apply_dmp.py <pfad_to_dmp(+.pkl)> <end_position x> <end_position y> <end_position z>

Bei der end_position muss man sich ein wenig spielen. Nicht alle End Positionen funktionieren gut.

Als Beispiel eignet sich aus der Home-Position z.B.:
python3 python_files/apply_dmp.py dmp/pick_2.pkl 0.2 0.2 0.05

Danach würde ich in die Startposition zurückfahren:
python3 python_files/apply_dmp.py dmp/return.pkl 0.0 0.0 0.3
(Es wird die Startposition nicht ganz erreicht, aber als Näherung reicht es)

Und anschließend eine neue Zielposition:
python3 python_files/apply_dmp.py dmp/pick_2.pkl 0.2 -0.15 0.05


Ihr werdet erkennen, das nicht alle Positionen erreicht werden, und das die Zielgenauigkeit sehr schlecht ist. Aber es funktioniert zumindest einigermaßen fürs erste. Womöglich sollten wir nochmal passende Trajetorien aufnehmen.

Zur Info: Als Startposition für die dmp wird immer die aktuelle Position verwendet. Dazu wird eine Message mit der aktuellen Joint-Position empfangen und anschließend mit der Vorwärts-Kinematik in kartesische Koordinaten umgerechnet.


