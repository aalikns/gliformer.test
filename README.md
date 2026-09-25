# GLiFormer ve Encoder-Decoder Mimarisi

Bu döküman, [huggingface.co/knowledgator/gliformer-large-v1](https://huggingface.co/knowledgator/gliformer-large-v1) sayfasında paylaşılan GLiFormer Large v1 modelinin nasıl çalıştığını, encoder-decoder mimarisini ve bu modelin büyük dil modellerinden (LLM) farkını anlatır.

## İçindekiler

1. [Encoder ve Decoder Nedir](#1-encoder-ve-decoder-nedir)
2. [Attention Mekanizması](#2-attention-mekanizması)
3. [DeBERTa Mimarisi ve Parametre Kavramı](#3-deberta-mimarisi-ve-parametre-kavramı)
4. [GLiFormer'ın Beş Görev Başlığı](#4-gliformerın-beş-görev-başlığı-head)
5. [Çoklu Görev Eğitimi](#5-çoklu-görev-eğitimi)
6. [Zero-Shot Kavramı](#6-zero-shot-kavramı)
7. [Değerlendirme Tablolarında Geçen Terimler](#7-değerlendirme-tablolarında-geçen-terimler)
8. [Modelin Görsel Analiz Yeteneği](#8-modelin-görsel-analiz-yeteneği)
9. [NLU ile LLM Arasındaki Fark](#9-nlu-ile-llm-arasındaki-fark)

---

## 1. Encoder ve Decoder Nedir

Bir metni işleyen bir model iki farklı işi yapabilir: metni anlamak ya da metnin devamını üretmek. Bu iki iş, farklı mimariler gerektirir.

**Encoder**, bir cümle verildiğinde, cümledeki her kelimeye cümlenin tamamına bakarak (hem öncesine hem sonrasına) bir anlam yükler. Buna çift yönlü (bidirectional) işleme denir.

**Decoder**, bir metnin devamını kelime kelime, sırayla üretir. Her yeni kelimeyi üretirken sadece kendinden önce gelen kelimeleri görebilir, henüz üretilmemiş olan kelimeleri göremez. Buna tek yönlü (unidirectional / causal) işleme denir.

Decoder'ın geleceği görememesi bilinçli bir tasarımdır. Eğitim sırasında decoder'a geleceği görme izni verilirse, model gerçek cevabı zaten görüp kopyalar, hiçbir şey öğrenmez. Bu yüzden decoder'ın attention hesabında, henüz üretilmemiş kelimelere karşılık gelen hücreler matematiksel olarak `-sonsuz` yapılır, bu da o hücrelerin ağırlığını otomatik olarak sıfırlar.

Bir decoder modeline (örneğin bir sohbet asistanına) bir mesaj yazıldığında, o mesaj modelin geçmişi sayılır, çünkü modelden önce yazılmıştır. Model cevap üretirken bu geçmişin tamamına bakabilir. Ayrı bir encoder gerekmez, çünkü mesaj ve cevap aynı, tek metin akışının parçasıdır. Bu mimari orijinal olarak çeviri gibi görevler için (kaynak dil ile hedef dil arasında) tasarlanmıştı; sohbet modelleri bu ihtiyaca sahip olmadığı için sadece decoder kısmını kullanır. Anlama görevleri için tasarlanan modeller (DeBERTa, BERT gibi) ise sadece encoder kısmını kullanır.

## 2. Attention Mekanizması

Attention, bir kelimenin cümledeki diğer kelimelere ne kadar önem vermesi gerektiğini hesaplayan mekanizmadır.

Bir cümledeki her kelime önce bir vektöre (sayı listesine) çevrilir. Bu vektörler gerçekte 768-1024 boyutlu olur, örnek olarak 3 boyutlu gösterilebilir:

```
Ali    = [0.2, 0.8, 0.1]
gitti  = [0.5, 0.3, 0.9]
okula  = [0.1, 0.6, 0.4]
```

"gitti" kelimesi işlenirken, diğer kelimelerle nokta çarpımı yapılır:

```
gitti · Ali   = (0.5×0.2)+(0.3×0.8)+(0.9×0.1) = 0.43
gitti · okula = (0.5×0.1)+(0.3×0.6)+(0.9×0.4) = 0.59
gitti · gitti = (0.5×0.5)+(0.3×0.3)+(0.9×0.9) = 1.15
```

Bu sayılar softmax işlemiyle yüzdeye çevrilir:

```
Ali'ye dikkat:    %20
gitti'ye dikkat:  %52
okula'ya dikkat:  %28
```

Bu yüzdelerle tüm kelime vektörlerinin ağırlıklı ortalaması alınır ve "gitti" kelimesinin yeni, bağlam içeren vektörü oluşur. Bir kelimenin kendine yüksek ağırlık vermesi beklenen bir durumdur, çünkü kelimenin kendi anlamı, onu işlerken en önemli bilgi kaynağıdır.

Encoder'da attention matrisinde her kelime her kelimeye bakabilir:

| | Ali | gitti | okula |
|---|---|---|---|
| **Ali** | evet | evet | evet |
| **gitti** | evet | evet | evet |
| **okula** | evet | evet | evet |

Decoder'da ise sadece geçmişe bakılabilir:

| | Ali | gitti | okula |
|---|---|---|---|
| **Ali** | evet | hayır | hayır |
| **gitti** | evet | evet | hayır |
| **okula** | evet | evet | evet |

"Hayır" işaretli hücreler hesaplamadan çıkarılır.

### Örnek: "Alice works at Acme" cümlesinde "Alice"in kişi olduğu nasıl hesaplanır

1. "Alice" kelimesi bir vektöre çevrilir.
2. Attention ile "works", "at", "Acme" kelimelerine bakarak yeni, bağlam içeren bir vektör oluşturulur.
3. Bu işlem model boyunca (DeBERTa Large'da yaklaşık 24 katman) tekrarlanır, her katmanda vektör biraz daha anlamlı hale gelir.
4. Son katmandan çıkan vektör, NER head'ine gönderilir.
5. NER head, küçük bir matrisle bu vektörü çarpıp dört sayı üretir: person, organization, location, none olasılıkları.
6. Bu matrisin içindeki sayılar, eğitim sırasında milyonlarca örnekten öğrenilmiştir. Cümle başında, büyük harfle başlayan, fiilden önce gelen kelimelerin genelde kişi olduğu gibi örüntüler bu sayılara işlenmiştir.
7. Sonuç olarak person olasılığı yüksek çıkar.

Bu süreçte hiçbir yerde "büyük harfle başlıyorsa kişidir" şeklinde yazılı bir kural yoktur. Bu tamamen istatistiksel bir örüntüdür.

Her katmanda attention bloğundan sonra bir de feed-forward bloğu (MLP - Multi-Layer Perceptron) bulunur:

```
Girdi (1024 sayı) -> matris (1024'ten 4096'ya genişlet) -> aktivasyon 
                   -> matris (4096'dan 1024'e daralt) -> Çıktı (1024 sayı)
```

Attention kelimeler arası ilişkiyi işlerken, feed-forward bloğu her kelimeyi ayrı ayrı, daha derin işler.

## 3. DeBERTa Mimarisi ve Parametre Kavramı

DeBERTa, katmanlar halinde üst üste dizilmiş bir yapıdır:

```
Girdi metni
    -> Embedding katmanı (kelimeleri ilk vektörlere çevirir)
    -> Katman 1 (Self-attention + Feed-forward)
    -> Katman 2
    -> ...
    -> Katman 24
    -> Son çıktı vektörleri
```

DeBERTa'yı BERT'ten ayıran özellik disentangled attention'dır: her kelimenin içeriği (anlamı) ve pozisyonu (cümledeki yeri) ayrı vektörlerde tutulur, attention hesaplanırken bu ikisi ayrı işlenip sonra birleştirilir.

## 4. GLiFormer'ın Beş Görev Başlığı (Head)

GLiFormer, tek bir DeBERTa encoder üzerine beş farklı görev başlığı eklenerek eğitilmiştir. Hepsi aynı encoder çıktısından beslenir, her biri bu çıktıyı farklı bir soruya çevirir.

### NER Head

NER, Named Entity Recognition (İsimlendirilmiş Varlık Tanıma) anlamına gelir. Her kelimeye tek tek, bağımsız bir karar verir.

```python
entities = model.predict_entities(
    "Alice works at Acme in London.",
    ["person", "organization", "location"],
    threshold=0.5,
)
```

Çıktı: Alice kişi, Acme organizasyon, London konum olarak etiketlenir. Her sonuç metin, etiket, başlangıç pozisyonu, bitiş pozisyonu ve güven skoru bilgisi içerir.

Threshold, bir tahmini geçerli saymak için gereken minimum güven skorudur (0 ile 1 arası).

### Classification Head

Tek tek kelimelere değil, cümlenin bütününe bakıp tek bir karar verir.

```python
predictions = model.classify(
    "The new search feature is fast and easy to use.",
    ["positive", "negative", "neutral"],
    threshold=0.5,
)
```

Adlandırılmış gruplar da desteklenir, örneğin duygu ve konu sınıflandırması tek çağrıda birlikte yapılabilir.

### Joint Relation Head

NER'in "bu nedir" sorusundan farklı olarak, iki varlığı birlikte değerlendirip aralarında belirli bir ilişki olup olmadığını sorar.

```python
results = model.inference(
    "Alice works at Acme.",
    joint_relations={
        "employment": {
            "entities": ["person", "organization"],
            "relations": ["works_at"],
        }
    },
    threshold=0.5,
)
```

Çıktı, `Alice - works_at - Acme` şeklinde bir üçlüdür.

Bu süreçte önce NER çalışır, Alice ve Acme'nin ne tür varlıklar olduğu tek tek belirlenir. Ardından bu bulunan varlıklar çift olarak alınıp aralarında ilişki aranır. NER, cümledeki bağlamı attention aracılığıyla zaten kullanır ama bunu resmi bir ilişki adı olarak çıktı vermez; sadece bir kelimenin organizasyon olduğunu söyler. İlişkiyi isimlendirip resmi bir üçlü üretmek, NER'in bulduklarının üzerine inşa edilen sonraki bir adımdır.

### Structuring Head

Bilgiyi önceden tanımlanmış bir şemaya yerleştirir.

```python
records = model.structure(
    "Alice works at Acme.",
    {"employee": ["name", "company"]},
)
# Çıktı: {'employee': [{'name': 'Alice', 'company': 'Acme'}]}
```

JSON, verileri anahtar-değer çiftleri halinde düzenleyen, hem insan tarafından okunabilen hem programlar tarafından kolayca işlenebilen bir formattır. Python'daki karşılığı dict veri tipidir. Serbest metni JSON'a çevirmenin nedeni, bir programın bu bilgiyi ayrıca ayrıştırmak zorunda kalmadan doğrudan kullanabilmesidir.

Nested (iç içe) şemalar da desteklenir. Şirket içinde departmanlar, departmanlar içinde çalışanlar gibi çok katmanlı yapılar Pydantic kütüphanesiyle tanımlanıp doldurulabilir.

### Embedding Head

Bir cümlenin anlamını sabit boyutlu bir sayı listesine çevirir.

```python
embeddings = model.embed_text([
    "A scientist works in a laboratory.",
    "A researcher conducts an experiment.",
])
print(embeddings.shape)  # torch.Size([2, 1024])
```

Her cümle 1024 sayıdan oluşan bir vektöre çevrilir. İki cümlenin vektörleri cosine similarity ile karşılaştırılarak anlamca ne kadar benzer oldukları ölçülebilir; sonuç 0 ile 1 arasında olup 1'e yakınlık aynı anlamı gösterir.

### Head Eklemenin Teknik Karşılığı

Bir head, encoder çıktısını alıp küçük bir ek matris zinciriyle spesifik bir karara çeviren yapıdır:

```
Encoder çıktısı (1024 sayı)
    -> matris (1024'ten 256'ya)
    -> aktivasyon
    -> matris (256'dan 4'e, 4 = kategori sayısı)
    -> softmax
    -> person: %2, organization: %94, location: %3, none: %1
```

Bu, encoder'a göre küçük bir ek yapıdır; her görev için ayrı bir tane eklenir.

## 5. Çoklu Görev Eğitimi

Eğitim sırasında farklı görevlere ait örnekler (NER, sınıflandırma, ilişki, yapılandırma) karışık sırayla gösterilir. Eğer önce sadece bir görevle, sonra sadece başka bir görevle eğitim yapılsaydı, sonraki aşamada encoder'ın ağırlıkları o göreve göre güncellenirken önceki görev için öğrenilenler bozulabilirdi. Bu duruma catastrophic forgetting (felaket unutması) denir. Örneklerin karıştırılması, encoder'ın hiçbir göreve aşırı özelleşip diğerini unutmamasını sağlar.

Bir NER örneği geldiğinde, sadece encoder ve NER head kullanılır, diğer head'ler o turda hiç devrede değildir ve hiçbir güncelleme almaz. Bir sınıflandırma örneği geldiğinde ise sadece encoder ve classification head güncellenir. Bu, backpropagation'ın (geri yayılım) matematiksel yapısından kaynaklanır; hata, sadece o turda kullanılan yol boyunca geriye yayılır. Encoder her turda (hangi görev olursa olsun) güncellenir çünkü her görev onu kullanır; head'ler ise sadece kendi görev türlerinde güncellenir.

## 6. Zero-Shot Kavramı

Zero-shot, modele göreve özel örnek göstermeden, sadece görevi ya da etiketi tanımlayıp sonuç almak anlamına gelir.

Bir büyük dil modeline "bu cümleyi sınıflandır" denildiğinde, bu talimat doğal dilde okunup anlaşılır ve serbest bir cevap üretilir. Model, eğitimi sırasında milyarlarca farklı talimat-cevap örneği gördüğü için, hiç görmediği yeni bir talimatı bile anlayıp uygun cevap üretebilir.

GLiFormer'da ise doğal dilde talimat yazılmaz, fonksiyon çağrısıyla ne istendiği belirtilir:

```python
model.classify(text, ["positive", "negative"])     # çalışır
model.classify(text, "bu cümlenin duygusunu bul")  # çalışmaz
```

GLiFormer'ın zero-shot özelliği, fonksiyon çerçevesinin (classify, predict_entities gibi) sabit kalması ama bu çerçevenin içindeki etiket isimlerinin değişken olması anlamına gelir. Model, önceden hiç görmediği bir etiket ismini bile anlayıp tahmin yapabilir, yeniden eğitilmesine gerek kalmadan.

Fark şu şekilde özetlenebilir: büyük dil modelleri hem görevin ne olduğunu hem nasıl yapılacağını serbest dilde anlar. GLiFormer'da görevin çerçevesi kod tarafında sabittir, model sadece etiket isminin anlamını çözebilir. Bu, büyük dil modellerindeki esneklikten çok daha dar bir zero-shot türüdür, ama eski nesil NER modellerinin tamamen sabit kategorilere kilitli olmasına kıyasla önemli bir gelişmedir.

README'de CrossNER gibi test sonuçları, modelin hiç görmediği alanlarda başarılı olduğunu gösterir şekilde sunulur. Ancak README, raporlanan transfer gruplarının o alanın eğitimde hiç görülmediğini garanti etmediğini belirtir. Model, test setinin kendisini görmemiş olabilir ama eğitim verisinde benzer bir konuya farklı bir veri setinden aşinalığı olabilir. Bu nedenle sonuçlar tam anlamıyla saf bir zero-shot performansı olarak değil, temkinli okunmalıdır.

## 7. Değerlendirme Tablolarında Geçen Terimler

**F1 skoru**, modelin doğru bulma oranı (precision) ile kaçırmama oranını (recall) tek sayıda özetleyen bir başarı ölçütüdür, 0 ile 100 arasında değer alır.

**Precision**, modelin "buldum" dediklerinin ne kadarının gerçekten doğru olduğunu gösterir.

**Recall**, gerçek varlıkların ne kadarının model tarafından kaçırılmadan bulunduğunu gösterir.

**Strict entity F1**, bir varlığı doğru saymak için hem sınırlarının hem tipinin tam doğru olması gerektiği sıkı bir ölçüm biçimidir.

**Macro-F1**, her sınıfa eşit ağırlık verilerek hesaplanan ortalama F1'dir. Örneğin pozitif sınıf 900 örnek ve 0.95 F1, negatif sınıf 80 örnek ve 0.60 F1, nötr sınıf 20 örnek ve 0.30 F1 değerine sahipse, macro-F1 bu üç değerin basit ortalamasıdır: (0.95+0.60+0.30)/3 = 0.617.

**Weighted-F1**, aynı örnekte her sınıfın F1'ini örnek sayısına göre ağırlıklandırır: (900×0.95 + 80×0.60 + 20×0.30)/1000 = 0.909. Weighted-F1 yüksek ama macro-F1 düşükse, bu modelin çoğunluk sınıfında iyi ama azınlık sınıflarında zayıf olduğu anlamına gelir.

İlişki çıkarma tablosundaki **"gold relations"**, insan eliyle etiketlenmiş, doğru kabul edilen referans veriyi ifade eder. Örneğin DocRED veri setinde gold relations 6003 ise, test setinde doğru kabul edilen toplam 6003 ilişki örneği vardır; bu, modelin bulması gereken hedef sayıdır.

DocRED satırında precision 29.90, recall 8.13, micro-F1 12.78, macro-F1 3.64 olarak raporlanmıştır. Bu, modelin 100 ilişki bulduğunu iddia ettiğinde bunların yaklaşık 30'unun doğru olduğunu, ancak gerçek ilişkilerin sadece yüzde 8'ini yakalayabildiğini gösterir. Macro-F1'in micro-F1'den çok daha düşük olması, modelin bazı nadir ilişki türlerinde neredeyse hiç başarılı olamadığını gösterir.

## 8. Modelin Görsel Analiz Yeteneği

Bu model görsel analiz yapamaz. README'nin sınırlamalar bölümü, bu checkpoint'in ayrı bir görsel (vision) başlığı olmadığını açıkça belirtir.

## 9. NLU ile LLM Arasındaki Fark

| | NLU (GLiFormer) | LLM (büyük dil modelleri) |
|---|---|---|
| **Temel yapı** | Encoder-only | Decoder-only |
| **Attention yönü** | Çift yönlü | Tek yönlü |
| **Yaptığı iş** | Var olan metni analiz eder, etiketler | Yeni metin üretir |
| **Çıktı türü** | Sabit formatlı | Serbest, akıcı metin |
| **Gereken geçiş sayısı** | Bir (tüm metin bir kerede işlenir) | Üretilen her kelime için bir geçiş |
| **Talimat anlama** | Yok, fonksiyon çağrısıyla yönlendirilir | Var, doğal dilde talimat okunup yorumlanır |
| **Sohbet edebilme** | Hayır | Evet |
| **Parametre boyutu** | Yüz milyonlar mertebesinde | Milyarlar - yüzlerce milyar mertebesinde |

NLU modelleri, milyonlarca cümlenin hızlı ve ucuz şekilde etiketlenmesi gerektiğinde, çıktının her zaman sabit ve garantili bir formatta olması gerektiğinde, düşük maliyetle yerel olarak çalıştırılmak istendiğinde ve belgelerden yapılandırılmış veri çekmek gibi dar, tekrarlı işlerde tercih edilir.

Büyük dil modelleri ise sohbet etme, soru cevaplama, açıklama yazma gibi görevlerde, görevin kendisinin de belirsiz olup talimatın anlaşılması gerektiği durumlarda, yaratıcı yazı ve karmaşık akıl yürütme gerektiren işlerde tercih edilir.

Bir büyük dil modeli, teorik olarak metinden varlık çıkarma gibi işleri de yapabilir, ancak her istek için nispeten yavaş bir üretim süreci gerektirir ve çıktı formatı her zaman garantili olmayabilir. Küçük bir encoder modeli aynı işi çok daha düşük maliyetle, hızlı ve garantili formatta yapar. Bu nedenle iki model türü birbirinin yerine geçmez, farklı iş türleri için tasarlanmıştır.

## Özet

GLiFormer, tek bir DeBERTa encoder üzerine beş farklı görev başlığı (NER, sınıflandırma, ilişki çıkarma, yapılandırma, embedding) eklenerek çoklu görev eğitimiyle oluşturulmuş bir modeldir. Encoder mimarisi, metni çift yönlü attention ile analiz eder; bu, onu tek yönlü çalışan ve metin üreten decoder tabanlı büyük dil modellerinden ayırır. Modelin zero-shot özelliği, önceden tanımlanmamış etiketleri de tanıyabilmesini sağlar, ancak bu, büyük dil modellerindeki serbest talimat anlama yeteneğinden farklı ve daha kısıtlı bir mekanizmadır. Değerlendirme sonuçları, modelin sınıflandırma ve yapılandırma görevlerinde güçlü, ilişki çıkarmada zayıf olduğunu ve tüm sonuçların İngilizce üzerinde raporlandığını göstermektedir.

---

*Hasan Ali Kınaş*
