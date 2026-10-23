import express, { Request, Response } from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT) || 3000;

  app.use(express.json());

  // Chatbot endpoint: Proxies requests to OpenRouter securely from the server
  app.post('/api/chat', async (req: Request, res: Response) => {
    try {
      const { messages, language = 'ru' } = req.body;

      if (!messages || !Array.isArray(messages)) {
        return res.status(400).json({ error: 'Messages array is required' });
      }

      const openRouterKey = process.env.OPENROUTER_API_KEY;
      const appUrl = process.env.APP_URL || `http://localhost:${PORT}`;

      const systemPrompt =
        language === 'kz'
          ? `Сіз — NeuroFactory AI өнеркәсіптік жасанды интеллект платформасының бас инженер-кеңесшісісіз.
Қазақстанның ауыр өнеркәсіп кәсіпорындарына (металлургия, мұнай-газ, тау-кен, химия) арналған шешімдерге жауаптысыз.
Сіздің мамандануыңыз:
1. SafeVision: ЖҚҚ (қорғаныс дулығасы, сигналдық кеудеше, қорғаныс көзілдірігі) сақталуын бақылау және қауіпті аймақтарға (кран аймағы, балқыту пештері) рұқсатсыз кіруді анықтау.
2. EcoPredict: Газ талдағыштардың (CO, NO2, SO2, PM2.5, PM10) көрсеткіштері, ҚР экологиялық нормалары (ШРК - шекті рұқсат етілген концентрация) және залпты шығарындыларды алдын ала болжау.
3. FailPredict: Жабдықтардың (турбокомпрессорлар, гидравликалық престер, диірмендер) діріл үдеуі (ISO 10816 бойынша RMS мм/с) мен подшипник температурасын талдау, істен шығуға дейінгі қалдық ресурс (RUL) және инженерлік нұсқаулықтар.
Міндетті түрде сауатты, нақты және кәсіби ҚАЗАҚ ТІЛІНДЕ жауап беріңіз. Инженерлік дәлдікпен, анық және түсінікті жазыңыз.`
          : `Вы — главный инженер-консультант промышленной платформы искусственного интеллекта NeuroFactory AI.
Вы отвечаете за индустриальные решения для тяжелой промышленности Казахстана и СНГ (металлургия, нефтехимия, горная добыча, машиностроение).
Ваша специализация:
1. SafeVision: Нейросетевой контроль СИЗ (каски, сигнальные жилеты, защитные очки) и предотвращение проникновения в опасные зоны (крановые пролеты, зоны плавильных печей).
2. EcoPredict: Экологический мониторинг газов (CO, NO2, SO2, пыль PM2.5/PM10), нормативы ПДК по стандартам Республики Казахстан и раннее предупреждение залповых выбросов.
3. FailPredict: Вибродиагностика агрегатов (турбокомпрессоры, насосы, клети) по стандарту ISO 10816 (RMS вибрация мм/с, температура подшипников), расчет остаточного ресурса RUL и формирование рекомендаций механику.
Отвечайте строго на РУССКОМ ЯЗЫКЕ. Предоставляйте точные, практические инженерные ответы с опорой на регламенты промбезопасности.`;

      // If Gemini API key is configured, invoke official @google/genai
      const geminiApiKey = process.env.GEMINI_API_KEY;
      if (geminiApiKey && geminiApiKey.trim() !== '') {
        try {
          const ai = new GoogleGenAI({ apiKey: geminiApiKey });
          const contents = messages.map((m: { sender: string; text: string }) => ({
            role: m.sender === 'user' ? 'user' : 'model',
            parts: [{ text: m.text }],
          }));

          const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents,
            config: {
              systemInstruction: systemPrompt,
              temperature: 0.3,
            },
          });

          const reply = response.text;
          if (reply) {
            return res.json({ reply });
          }
        } catch (geminiErr) {
          console.error('Error invoking Gemini API:', geminiErr);
        }
      }

      // If OpenRouter API key is configured, invoke OpenRouter
      if (openRouterKey && openRouterKey.trim() !== '') {
        try {
          const apiPayload = {
            model: 'meta-llama/llama-3.3-70b-instruct',
            messages: [
              { role: 'system', content: systemPrompt },
              ...messages.map((m: { sender: string; text: string }) => ({
                role: m.sender === 'user' ? 'user' : 'assistant',
                content: m.text,
              })),
            ],
            temperature: 0.3,
            max_tokens: 1000,
          };

          const openRouterRes = await fetch('https://openrouter.ai/api/v1/chat/completions', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${openRouterKey}`,
              'HTTP-Referer': appUrl,
              'X-Title': 'NeuroFactory AI Platform',
            },
            body: JSON.stringify(apiPayload),
          });

          if (openRouterRes.ok) {
            const data = await openRouterRes.json();
            const reply = data.choices?.[0]?.message?.content;
            if (reply) {
              return res.json({ reply });
            }
          } else {
            console.warn('OpenRouter API returned non-200 status:', openRouterRes.status);
          }
        } catch (openRouterErr) {
          console.error('Error invoking OpenRouter API:', openRouterErr);
        }
      }

      // Intelligent Industrial AI Domain Knowledge Engine fallback
      const lastUserMsg = messages[messages.length - 1]?.text || '';
      const reply = generateDomainAIResponse(lastUserMsg, language);
      return res.json({ reply });
    } catch (err: unknown) {
      console.error('Chat endpoint error:', err);
      res.status(500).json({ error: 'Internal server error processing chat request' });
    }
  });

  // Health check endpoint
  app.get('/api/health', (req, res) => {
    res.json({
      status: 'nominal',
      service: 'NeuroFactory Industrial Core',
      timestamp: new Date().toISOString(),
    });
  });

  // Serve static build in production, or mount Vite middleware in development
  if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`NeuroFactory AI full-stack server running at http://0.0.0.0:${PORT}`);
  });
}

/**
 * Domain-specific industrial AI logic engine for NeuroFactory
 */
function generateDomainAIResponse(question: string, language: 'ru' | 'kz'): string {
  const q = question.toLowerCase();

  if (language === 'kz') {
    if (q.includes('safevision') || q.includes('жққ') || q.includes('дулыға') || q.includes('көзілдірік') || q.includes('аймақ')) {
      return `**SafeVision модульі бойынша инженерлік түсініктеме:**
1. **ЖҚҚ анықтау:** Жүйе YOLOv11 Edge нейрожелісін қолданады. Каска мен сигналдық кеудешені анықтау дәлдігі 94.2%-ды құрайды.
2. **Қорғаныс көзілдірігі:** Шлифовка және дәнекерлеу бекеттерінде тұлғаны сегментациялау арқылы көзілдірік (goggles) бар-жоғы тексеріледі.
3. **Қауіпті аймақтар:** Кран жүру жолдары мен балқыту аймақтарына персонал кірген кезде 0.3 секунд ішінде ескерту жасалып, цех диспетчеріне белгі беріледі.
4. **Сүзгілер:** Барлық бұзушылықтар журналға уақыты, учаскесі және бейне-дәлелімен жазылады.`;
    }

    if (q.includes('ecopredict') || q.includes('шрк') || q.includes('so2') || q.includes('co') || q.includes('шығарынды') || q.includes('экология')) {
      return `**EcoPredict экологиялық мониторинг бойынша мәлімет:**
1. **ШРК нормативтері (ҚР заңнамасы):**
   - **SO₂ (Күкірт диоксиді):** ШРК тәуліктік орташа нормасы — 0.5 мг/м³, максималды бір реттік — 0.05 мг/м³.
   - **CO (Көміртек оксиді):** ШРК — 3.0–5.0 мг/м³.
   - **PM2.5 / PM10 (Шаң):** 0.15 мг/м³ аспауы тиіс.
2. **Болжау моделі:** LSTM уақыттық қатарлары ауа райы (жел бағыты, инверсия) мен пеш жүктемесін ескере отырып, 24 сағат бұрын мүмкін болатын шығарынды шыңдарын алдын ала болжайды.`;
    }

    if (q.includes('failpredict') || q.includes('діріл') || q.includes('вибрация') || q.includes('ақау') || q.includes('компрессор') || q.includes('rul')) {
      return `**FailPredict вибродиагностика және RUL талдауы:**
1. **ISO 10816 стандарты:**
   - **< 2.3 мм/с:** Тамаша жағдай (А тобы).
   - **2.3 – 4.5 мм/с:** Қанағаттанарлық жағдай (В тобы).
   - **4.5 – 7.1 мм/с:** Ескерту қажет, дефекттің басталуы (С тобы).
   - **> 7.1 мм/с:** Қауіпті деңгей (D тобы). Агрегатты шұғыл тоқтату ұсынылады.
2. **Ұсыным:** 6.8 мм/с кезінде қозғалтқыш пен редуктор подшипниктерін (SKF 22218) жедел майлау және осін тексеруге наряд-рұқсаттама қалыптастырыңыз.`;
    }

    return `**NeuroFactory AI жүйесінің жауабы:**
Сіздің сұрағыңыз қабылданды. Платформа 3 бағытта қызмет атқарады:
- **SafeVision:** Еңбекті қорғау және қауіпсіздік техникасы (дулыға, кеудеше, қауіпті аймақ).
- **EcoPredict:** Ауаға шығарылатын газдарды болжау және ШРК нормаларын бақылау.
- **FailPredict:** Жабдықтардың дірілі мен қызуын диагностикалау, істен шығу қаупін болжау.
Қосымша нақты сұрақтарыңыз болса, сұраңыз!`;
  }

  // Russian response
  if (q.includes('safevision') || q.includes('каск') || q.includes('жилет') || q.includes('очки') || q.includes('сиз') || q.includes('зон')) {
    return `**Инженерная справка по модулю SafeVision:**
1. **Контроль СИЗ:** Используется модель YOLOv11 Edge с инференсом до 25 FPS на камеру. Распознаются каски (желтые, белые, синие), сигнальные жилеты со светоотражающими полосами и защитные поликарбонатные очки.
2. **Опасные зоны:** Периметр кран-балок и разливочных ковшей задается полигональными масками. При пересечении границы срабатывает тревога в течение 0.3 секунды.
3. **Верификация:** В журнале каждое событие сопровождается статусом ("Новое", "Подтверждено", "Ложное"). Инженер по охране труда может мгновенно закрыть инцидент или направить предписание.`;
  }

  if (q.includes('ecopredict') || q.includes('пдк') || q.includes('so2') || q.includes('co') || q.includes('выброс') || q.includes('эколог')) {
    return `**Инженерная справка по модулю EcoPredict:**
1. **Нормативы ПДК (Республика Казахстан):**
   - **SO₂ (Диоксид серы):** ПДК м.р. — 0.5 мг/м³, среднесуточная — 0.05 мг/м³.
   - **CO (Оксид углерода):** ПДК м.р. — 5.0 мг/м³.
   - **Пыль неорганическая (PM2.5 / PM10):** 0.15 мг/м³.
2. **Предиктивный алгоритм:** Рекуррентные сети LSTM прогнозируют накопление концентраций на 24 часа вперед с учетом розы ветров и температурной инверсии. Это позволяет снизить нагрузку на печи до фиксации превышения госорганами.`;
  }

  if (q.includes('failpredict') || q.includes('вибраци') || q.includes('rul') || q.includes('компрессор') || q.includes('подшипник') || q.includes('наряд')) {
    return `**Инженерная справка по модулю FailPredict:**
1. **Стандарт вибрации ISO 10816-3:**
   - **< 2.3 мм/с RMS:** Зона A (Отличное состояние).
   - **2.3 – 4.5 мм/с RMS:** Зона B (Допустимая эксплуатация без ограничений).
   - **4.5 – 7.1 мм/с RMS:** Зона C (Предупреждение: развитие дефекта подшипника или дисбаланса).
   - **> 7.1 мм/с RMS:** Зона D (Недопустимо: высокий риск заклинивания, аварийный останов).
2. **RUL (Remaining Useful Life):** При вероятности отказа свыше 70% рекомендуется сформировать наряд-допуск на вибродиагностику спектра 1X/2X и замену смазки в редукторе.`;
  }

  return `**Ответ аналитического центра NeuroFactory AI:**
Платформа осуществляет сквозной мониторинг промышленного предприятия по 3 ключевым модулям:
1. **SafeVision:** Защита персонала и детекция СИЗ на камерах цеха.
2. **EcoPredict:** Контроль ПДК вредных выбросов и предупреждение штрафов экологии.
3. **FailPredict:** Мониторинг виброакустики и температуры критических агрегатов.
Задайте уточняющий вопрос по интересующему оборудованию или технологическому участку!`;
}

startServer();
