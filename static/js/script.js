const species_data  = [
    {
        "name": "Indian Elephant",
        "characteristics": {
            "scientific_name": "Elephas maximus",
            "lifespan": "60-70 Yrs",
            "body": "11.3 ft.",
            "weight": "3,600 - 5000 kg",
            "height": "7 - 12 ft.",
            "info": "Indian elephants are found in dense forest regions including tropic forests of northeast, south and central India as well as the Sub-Himalayan region. Indian elephants inhabit savannas, marshes, and open grasslands, among other habitats. In addition, they are prevalent in hilly areas, water sources, and rainforests."
        }
    },
    {
        "name": "Indian Wolf",
        "characteristics": {
            "scientific_name": "Canis lupus pallipes",
            "lifespan": "12 - 16 Yrs",
            "body": "100 - 145 cm",
            "weight": "17 - 25 kg",
            "height": "57 - 72 cm",
            "info": "Indian wolf is the most endangered subspecies of Grey wolf. Its shorter fur is greyish intermingled with black on the dorsal crest while the underside is buff. Indian wolves inhabit the dry open country, deserts, and barren uplands. It's a social species that lives in a pack of 6-8 animals and communicates by howling and using gestures involving ears tails and facial muscles."
        }
    },
    {
        "name": "White Tiger",
        "characteristics": {
            "scientific_name": "Panthera Tigris Tigris",
            "lifespan": "8 - 10 Yrs",
            "body": "5 - 6 ft.",
            "weight": "135 - 230 kg",
            "height": "90 - 100 cm",
            "info": "It is a white-colored variant of the Bengal tiger, not a different species. The white color is due to a lack of pigment pheomelanin, the one found in tigers with orange color fur which is the result of genetic mutations. The first white tiger in India was spotted by Maharaja of Rewa who capture it and domesticated it to its death."
        }
    },
    {
        "name": "Asiatic Lion",
        "characteristics": {
            "scientific_name": "Panthera leo persica",
            "lifespan": "16 - 18 Yrs",
            "body": "115 inches",
            "weight": "160 - 190 kg",
            "height": "110 cm",
            "info": "Globally Asiatic lions are found only in India in Gir wildlife sanctuary across Gujrat state. It lives in small pride comprising 2-5 females with their young. Males join in only to eat and mate. Female lions are the pride's primary hunters & work together to prey upon large mammals."
        }
    },
    {
        "name": "Barking Deer",
        "characteristics": {
            "scientific_name": "Muntiacus muntjac",
            "lifespan": "20 - 30 Yrs",
            "body": "3 - 4 ft.",
            "weight": "20 - 30 kg",
            "height": "50 - 70 cm",
            "info": "Barking deer or rib-faced deer are also referred to as Muntjacs. It is a small deer of the genus Muntiacus that is native to South and Southeast Asia. The coat of the barking deer is short, soft, thick and dense, especially for those who live in colder climates. Depending on the season, the coat's colour can change from dark brown to yellowish and greyish brown."
        }
    },
    {
        "name": "Blackbuck",
        "characteristics": {
            "scientific_name": "Antelope Cervicapra",
            "lifespan": "10 - 15 Yrs",
            "body": "120 cm",
            "weight": "32 - 43 kg",
            "height": "73 - 83 cm",
            "info" : "Blackbuck is the antelope found exclusively in the Indian subcontinent and prefers arid grassland, open scrub and semi-desert areas. They live in the group of 5 to 50 with mostly single adult males and several adult females. Their sense of smell and hearing are not well developed and rely on eyesight for detecting danger."
        }
    },
    {
        "name": "Indian Bison",
        "characteristics": {
            "scientific_name": "Bos Gaurus",
            "lifespan": "26 - 30 Yrs",
            "weight": "840 kg",
            "height": "168 - 188 cm",
            "info" : "Gaur is also known as the Indian Bison. It is native to South Asia and Southeast Asia. The gaur is a bovid animal with a bulging forehead and a curved shape to its head. It has large ears and a bumpy ridge on its back. Adult males are dark brown and turn black as they get old. Their smooth, shiny fur is short, and they have pointed hooves."
        }
    },
    {
        "name": "Indian Rock Python",
        "characteristics": {
            "scientific_name": "Python molurus",
            "body": "Upto 18ft",
            "info" : "The Indian python is a large python species native to tropical and subtropical regions of the Indian subcontinent and Southeast Asia. It is also known by the common names black-tailed python, Indian rock python, and Asian rock python."
        }
    },
    {
        "name": "Lion-Tailed macaque",
        "characteristics": {
            "scientific_name": "Macaca silenus",
            "lifespan": "20 Yrs",
            "weight": "2 - 10 kg",
            "height": "40 to 61 cm",
            "info" : "Lion-tailed macaques are covered in black fur, and have a striking gray or silver mane that surrounds their face which can be found in both sexes. The face itself is hairless and black, being pinkish in infants less than a year old. They are named not for their mane, but for their tail, which is long, thin, and naked, with a lion-like, black tail tuft at the tip. The size of their tail is about 25 cm (9.8 in) in length. "
        }
    },
    {
        "name": "Nilgiri Tahr",
        "characteristics": {
            "scientific_name": "Nilgiritragus hylocrius",
            "lifespan": "17 Yrs",
            "weight": "50 - 100 kg",
            "info" : "The Nilgiri tahr can be found only in India. It inhabits the open montane grassland habitat of the South Western Ghats montane rain forests ecoregion. At elevations from 1,200 to 2,600 m (3,900 to 8,500 ft), the forests open into large grasslands interspersed with pockets of stunted forests, locally known as sholas. These grassland habitats are surrounded by dense forests at the lower elevations. The Nilgiri tahrs formerly ranged over these grasslands in large herds, but hunting and poaching in the 19th century reduced their population."
        }
    },
    {
        "name": "One Horned Rhino",
        "characteristics": {
            "scientific_name": "Rhinoceros unicornis",
            "lifespan": "30 -45 Yrs",
            "body": "6.6 - 8.2 ft long",
            "weight": "2200 kg",
            "height": "1.6 - 1.9m",
            "info": "The Indian rhinoceros, also known as the greater one-horned rhinoceros, great Indian rhinoceros, or Indian rhino for short, is a rhinoceros species native to the Indian subcontinent. It is the second largest extant species of rhinoceros, with adult males weighing 2.2 tonnes and adult females 1.6 tonnes."
        }
    },
    {
        "name": "Sambar Deer",
        "characteristics": {
            "scientific_name": "Cervix Unicolour",
            "lifespan": "20 -26 Yrs",
            "body": "5.3  - 8 ft long",
            "weight": "100 - 350 kg",
            "height": "102 - 106 cm",
            "info": "The sambar is a large deer that is widespread in South China, Southeast Asia, and the Indian subcontinent and is categorize as a vulnerable species. Populations have significantly decreased as a result of intense hunting, local rebellion, and industrial habitat exploitation. Sambar deer are light brown or dark with a greyish or yellowish tinge. The underparts are paler. Old sambars turn very dark brown, almost the colour black."
        }
    },
    {
        "name": "Sloth Bear",
        "characteristics": {
            "scientific_name": "Melursus ursinus",
            "lifespan": "Upto 40 Yrs",
            "body": "140 - 170 cm",
            "weight": "80 - 125 kg",
            "info": "This widespread Indian bear was earlier used as a performing bear on the streets. Their long, curved claws are used for penetrating termite mounds. The lack of missing front incisors facilitates to suck of termites and ants."
        }
    },
    {
        "name": "Indian Star Tortoise",
        "characteristics": {
            "scientific_name": "Geochelone Elegans",
            "info": "The Indian star tortoise is a threatened tortoise species native to India, Pakistan and Sri Lank where it inhabits dry areas and scrub forest. It has been listed as Vulnerable on the IUCN Red List since 2016, as the population is thought to comprise more than 10,000 individuals, but with a declining trend."
        }
    }
]

document.addEventListener("DOMContentLoaded", function () {
    const video = document.getElementById('camera');
    let videoStream;

    function startCamera() {
        const constraints = {
            video: {
                facingMode: 'environment'
            }
        };

        navigator.mediaDevices.getUserMedia(constraints)
            .then((stream) => {
                videoStream = stream;
                video.srcObject = videoStream;
            })
            .catch((error) => {
                console.error('Error accessing camera: ', error);
            });
    }

    function stopCamera() {
        if (videoStream) {
            const tracks = videoStream.getTracks();
            tracks.forEach(track => track.stop());
            videoStream = null;
        }
    }

    function captureImage() {
        const canvas = document.createElement("canvas");
        const camera = document.getElementById("camera");

        canvas.width = camera.videoWidth;
        canvas.height = camera.videoHeight;

        const context = canvas.getContext("2d");
        context.drawImage(camera, 0, 0, canvas.width, canvas.height);
        const imageDataURL = canvas.toDataURL("image/jpeg");

        const previewImage = document.getElementById("preview-image");
        previewImage.src = imageDataURL;
        previewImage.alt = "Captured Image";
        document.getElementById("preview-section").style.display = "block";
        previewImage.scrollIntoView({ behavior: 'smooth' });

        previewFile = dataURLtoFile(imageDataURL, "captured-image.jpg");

        stopCamera();
    }

    const form1 = document.querySelector('form[name="form1"]');
    form1.addEventListener('submit', function (event) {
        event.preventDefault();
        identifySpecies();
    });

    function dataURLtoFile(dataURL, filename) {
        const byteString = atob(dataURL.split(",")[1]);
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }
        const blob = new Blob([ab], { type: "image/jpeg" });
        return new File([blob], filename, { type: "image/jpeg" });
    }

    window.startCamera = startCamera;
    window.stopCamera = stopCamera;
    window.captureImage = captureImage;
});

function resetApp() {
    location.reload()
}

function handleFileUpload(file) {
    const reader = new FileReader();

    reader.onload = function (e) {
        document.getElementById('preview-image').src = e.target.result;
        previewImage.scrollIntoView({ behavior: 'smooth' });
    };

    reader.readAsDataURL(file);

}

function uploadMedia() {
    const input = document.getElementById("uploadInput");
    const file = input.files[0];

    previewFile = file;

    const reader = new FileReader();
    reader.onload = function (e) {
        const previewImage = document.getElementById("preview-image");
        previewImage.src = e.target.result;
        previewImage.alt = file.name;
        document.getElementById("preview-section").style.display = "block";
    };
    reader.readAsDataURL(file);
}

function identifySpecies() {
    if (!previewFile) {
        console.error("No file selected for preview image");
        return;
    }

    const formData = new FormData();
    formData.append("file", previewFile);

    console.log("formData:", formData);

    $.ajax({
        url: "/identify",
        type: "POST",
        data: formData,
        contentType: false,
        processData: false,
        success: function (response) {
            console.log("response:", response);
            search(response.label.toString())
        },
        error: function (xhr, textStatus, errorThrown) {
            console.error("xhr:", xhr);
            console.error("textStatus:", textStatus);
            console.error("errorThrown:", errorThrown);
        }
    });
}

function search(species) {
    let res = false
    if (species != NaN) {
        $.ajax({
            method: 'GET',
            url: 'https://api.api-ninjas.com/v1/animals?name=' + species.toString(),
            headers: { 'X-Api-Key': 'yUUcMvgCc5K84gXqrWTTyw==PXB6CQuS3FDzkO7h' },
            contentType: 'application/json',
            success: function (result) {
                console.log(result)

                species_data.forEach(function (animal) {
                    document.getElementById('processed-section').scrollIntoView({ behavior: 'smooth' });
                    if (animal.name.toLowerCase() == species.toLowerCase()) {
                        res = true
                        document.getElementById('processed-info').innerHTML =
                            '<p style="font-size=16px;">Characteristics</p>' +
                            "Scientific Name: " + animal.characteristics.scientific_name +
                            "<br><br>Lifespan: " + animal.characteristics.lifespan +
                            "<br><br>Body: " + animal.characteristics.body +
                            "<br><br>Weight: " + animal.characteristics.weight +
                            "<br><br>" + animal.characteristics.info
                    }
                });

                if(!res) {
                    result.forEach(function (animal) {
                        document.getElementById('processed-section').scrollIntoView({ behavior: 'smooth' });
                        if (animal.name.toLowerCase() == species.toLowerCase()) {
                            res = true
                            document.getElementById('processed-info').innerHTML =
                                '<p style="font-size=16px;">Characteristics</p>' +
                                "Scientific Name: " + animal.taxonomy.scientific_name +
                                "<br><br>Lifespan: " + animal.characteristics.lifespan +
                                "<br><br>Weight: " + animal.characteristics.weight +
                                "<br><br>Diet: " + animal.characteristics.diet +
                                "<br><br>" + animal.characteristics.slogan
                        }
                    });
                }

                if (!res) {
                    document.getElementById('processed-info').innerHTML =
                    '<p style="font-size=16px;">Not Found :/</p>'
                }

                document.getElementById('processed-section').style.display = 'block'
                document.getElementById('processed-text').textContent = species.toUpperCase()

            },
            error: function ajaxError(jqXHR) {
                console.error('Error: ', jqXHR.responseText);
            }
        });
    } else {
        alert('Please type/upload a wildlife species!')
    }
}