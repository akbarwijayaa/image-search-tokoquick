# <div align="left">API Documentation</div>

## <div align="left">Base URL</div>

```
http://ai.quick.com/
```
Seluruh jenis request menuju API menggunakan base URL tersebut.

<br>

## <div align="left">Endpoints</div>

- ###  Get API Info
    Endpoint ini digunakan untuk mendapatkan informasi bahwa API telah aktif.
    <br>

    **Endpoint**
    ```bash
    GET   /predict
    ```

    **Response**
    ```
    API Image Search Tokoquick
    ```
    <br>

- ### Image Search Engine
    Endpoint ini digunakan untuk melakukan query by image. User melakukan pencarian menggunakan gambar, kemudian sistem mengembalikan gambar-gambar dalam database yang memiliki fitur serupa dengan gambar query.
    <br>

    **Endpoint**
    ```bash
    POST   /predict
    ```
    **Request Body** `(form-data)` :
    * **`image`** _(files, required)_ : file gambar sebagai query.

    **Example Response**
    ```json
    {
    {
        "preds": [
            {
                "id_product": 6145,
                "product_name": "Wipro Bor Listrik SDS W6261 26mm",
                "score": 98.39
            },
            {
                "id_product": 6144,
                "product_name": "Wipro Bor Listrik SDS W6240 24mm",
                "score": 77.91
            },
            {
                "id_product": 6143,
                "product_name": "Wipro Bor Listrik Impact W6137 13mm",
                "score": 77.86
            },

            ...
            
        ],
        "success": true,
        "time": 0.67
    }
    }
    ```
    <br>

## <div align="left">Error Handling</div>
Object Countings API menggunakan standar HTTP status code sebagai indikasi sukses/gagal sebuah request.

* **200** _OK_

* **400** _bad request_

* **403** _forbidden_

* **404** _notfound_

* **405** _method not allowed_

* **408** _request timeout_

* **500** _internal server error_

* **502** _bad gateway_

* **503** _service unavailable_

* **504** _gateway timeout_
