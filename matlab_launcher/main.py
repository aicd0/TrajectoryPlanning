import matlab.engine

print('Starting session...')
eng = matlab.engine.start_matlab()
eng.on_startup(nargout=0)

print('Session name: ' + eng.workspace['engine_name'])
input('Press any key to stop the session...')

eng.quit()
print('Session stopped.')