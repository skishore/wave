def Settings(**kwargs):
    if kwargs[ 'language' ] == 'cfamily':
        return {'flags': ['-x', 'c++', '-std=c++17', '-Wall', '-Wconversion', '-Werror', '-Wno-sign-conversion', '-Wno-unknown-attributes']}
