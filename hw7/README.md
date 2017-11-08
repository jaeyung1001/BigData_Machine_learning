## 값 변환

frame = frame.set_value(frame[frame.D=='I'].index,'D','1')<br>
frame = frame.set_value(frame[frame.D=='F'].index,'D','0')