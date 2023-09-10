# Proyecto_PreTAWS_G4

# Instalar requirements

`pip install -r requirements.txt`

# En app.py

**1. Ingresar el mensaje en `mensaje`**

```python
if __name__ == '__main__':
    # main()

    '''
    1. Escribir el mensaje
    '''
    mensaje = 'A veces no se cómo sentirme cuándo no sale lo que quiero como lo quiero'
    # Obtener 2 párrafos coincidentes al mensaje
    file_name = 'matched_paragraphs_ex1'
    avance(mensaje, file_name, k=2)

    with open(f'./{file_name}.json', 'r') as f:
        matched = json.load(f)

    '''
    2. Leer el archivo json generado. Este contiene los párrafos que coinciden con el texto dado en {mensaje}
    '''
    ''' file_name.json 
    {'mensaje':
        [
            {
                'podcast':'nombre_del_podcast',
                'title':'titulo_del_episodio_del_podcast',
                'matched_paragraphs':['texto_primer_parrafo', 
                                        'texto_segundo_parrafo',
                                          ...,
                                            'texto_k-ésimo párrafo']
            },
            {...},
            ...,
            {...}
        ]
    }
    '''
    print(matched[f'{mensaje}'][0]['matched_paragraphs'])
```
**2. Ejecutar app.py**
Ejecutar en el terminal al archivo app.py

`python app.py`

**3. Leer el archivo json generado**
Este archivo contiene los párrafos que tienen similitud con el mensaje propuesto en `mensaje`.
Este archivo tiene la siguiente estructura con k-párrafos.
```javascript
{'mensaje':
        [
            {
                'podcast':'nombre_del_podcast',
                'title':'titulo_del_episodio_del_podcast',
                'matched_paragraphs':['texto_primer_parrafo', 
                                        'texto_segundo_parrafo',
                                          ...
                                    ]
            },
            {...},
            ...,
            {...}
        ]
    }
```