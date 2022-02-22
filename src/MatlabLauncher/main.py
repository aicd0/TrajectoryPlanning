import config
import matlab.engine
import os
import utils.fileio
import utils.string_utils

def main():
    print('Starting MATLAB...')
    eng = matlab.engine.start_matlab()
    eng.workspace['target_dir'] = utils.string_utils.to_folder_path(config.TargetDirectory)
    eng.onStartup(nargout=0)
    
    file_path = config.SessionFile
    folder_path = utils.string_utils.to_parent_path(file_path)
    utils.fileio.mktree(folder_path)
    with open(file_path, 'wb') as f:
        f.write(eng.workspace['engine_name'].encode('utf-8'))

    # Suspend the thread.
    input('Press any key to terminate...')

    os.remove(file_path)
    eng.quit()
    print('Session closed.')

if __name__ == '__main__':
    main()